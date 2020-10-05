import pyarrow
from pathlib import Path
from transformers import BertTokenizer, BertModel, BertConfig, BertForTokenClassification, pipeline, \
    AutoModelForTokenClassification, AutoTokenizer
import torch
import nltk
from nltk.corpus import stopwords
import string
from collections import Counter
import numpy as np
import pandas as pd
import argparse
import os
import random


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_output_folder", type=str,
                        help="Path to the inference folder containing the merged parquet file.")
    parser.add_argument("--tweets_to_label_output_path", type=str,
                        help="Path to the folder where the tweets to label are saved.")
    parser.add_argument("--nb_top_lift_kw", type=int,
                        help="Number of top-lift keywords to keep for MLM (Keyword exploration).")
    parser.add_argument("--nb_tweets_per_kw_mlm",
                        help="Number of tweets to use per keyword to do MLM (Keyword exploration).",
                        default="all")
    parser.add_argument("--nb_kw_per_tweet_mlm", type=int,
                        help="Number of keywords to extract from MLM for each tweet (Keyword exploration). ",
                        default=10)
    parser.add_argument("--nb_tweets_per_kw_final", type=int,
                        help="Number of tweets to send to labelling per keyword outputted by MLM (Keyword exploration).")
    parser.add_argument("--bootstrapping_method", type=str,
                        help="whether to bootstrap the MLM results of all keywords together or do it separately for each kw")
    parser.add_argument("--nb_bootstrapped_samples", type=int, help="Number of bootstrapped samples ", default=1000)
    parser.add_argument("--nb_final_candidate_kw", type=int,
                        help="Number of top keywords in terms of wordcount that are kept at the end of the process.",
                        default=10)
    parser.add_argument("--final_keywords_folder_path", type=int,
                        help="Number of top keywords in terms of wordcount that are kept at the end of the process.",
                        default=10)

    args = parser.parse_args()
    return args


def drop_stopwords_punctuation(df):
    """
    Drop rows containing stopwords or punctuation in the word column of the input dataframe.
    :param df: input pandas DataFrame, must contain a word column
    :return: pandas DataFrame without rows containing stopwords or punctuation
    """
    punctuation_list = [i for i in string.punctuation]
    all_stops = stopwords.words('english') + punctuation_list + ['[UNK]']
    df = df[~df['word'].isin(all_stops)].reset_index(drop=True)
    return df


def calculate_lift(top_df, nb_top_lift_kw):
    """
    Calculate keyword lift for each word appearing in top tweets.
    To do so, calculate word count for top tweets, then join with word count from random set and calculate lift.
    :param top_df: pandas DataFrame containing top tweets (based on the base rate estimate)
    :param nb_top_lift_kw: number of high lift keywords to retain (
    :return: either all keywords with lift > 1 if there is less than nb_top_lift_kw keywords
    or the top nb_top_lift_kw keywords with lift > 1.
    """
    top_wordcount_df = top_df.explode('convbert_tokenized_text')
    top_wordcount_df = top_wordcount_df['convbert_tokenized_text'].value_counts().rename_axis(
        'word').reset_index(name='count_top_tweets')
    full_random_wordcount_df = pd.read_parquet(
        '/scratch/mt4493/twitter_labor/twitter-labor-data/data/wordcount_random/wordcount_random.parquet')
    wordcount_df = top_wordcount_df.merge(full_random_wordcount_df, on=['word'])
    wordcount_df = drop_stopwords_punctuation(wordcount_df)
    wordcount_df['lift'] = (wordcount_df['count_top_tweets'] / wordcount_df[
        'count']) * N_random / label2rank[column]
    wordcount_df = wordcount_df.sort_values(by=["lift"], ascending=False).reset_index(drop=True)
    # Keep only word with lift > 1
    wordcount_df = wordcount_df.loc[wordcount_df['lift'] > 1].reset_index(drop=True)
    keywords_with_lift_higher_1_list = list(wordcount_df['word'].values)
    if wordcount_df.shape[0] < nb_top_lift_kw:
        return wordcount_df['word'].tolist(), keywords_with_lift_higher_1_list, full_random_wordcount_df
    else:
        return wordcount_df['word'][
               :nb_top_lift_kw].tolist(), keywords_with_lift_higher_1_list, full_random_wordcount_df


def sample_tweets_containing_selected_keywords(keyword, nb_tweets_per_keyword, data_df, lowercase, random):
    """
    Identify tweets containing a certain keyword and take either the top nb_tweets_per_keyword tweets in terms of score
    or a random sample of nb_tweets_per_keywords tweets containing this keyword.
    :param keyword: keyword to look for in tweets
    :param nb_tweets_per_keyword: the number of tweets to retain for one keyword
    :param data_df: the pandas DataFrame containing the tweets. Must have the columns: text, lowercased_text and score
    :param lowercase: whether to match a lowercased keyword with a lowercased tweet (if True) or not
    :param random: whether to take a random sample (if True) or top tweets
    :return: a pandas DataFrame with nb_tweets_per_keyword tweets or less containing the keyword
    """
    if not lowercase:
        tweets_containing_keyword_df = data_df[data_df['text'].str.contains(keyword)].reset_index(drop=True)
    else:
        tweets_containing_keyword_df = data_df[data_df['lowercased_text'].str.contains(keyword.lower())].reset_index(
            drop=True)
    tweets_containing_keyword_df = tweets_containing_keyword_df.sort_values(by=["score"], ascending=False).reset_index(
        drop=True)
    if nb_tweets_per_keyword == "all":
        return tweets_containing_keyword_df
    elif tweets_containing_keyword_df.shape[0] < nb_tweets_per_keyword:
        print("Only {} tweets containing keyword {} (< {}). Sending all of them to labelling.".format(
            str(tweets_containing_keyword_df.shape[0]), keyword, str(nb_tweets_per_keyword)))
        return tweets_containing_keyword_df
    else:
        if not random:
            return tweets_containing_keyword_df[:nb_tweets_per_keyword]
        elif random:
            return tweets_containing_keyword_df.sample(n=nb_tweets_per_keyword).reset_index(drop=True)


def extract_keywords_from_mlm_results(mlm_results_list, nb_keywords_per_tweet):
    """
    Extract keywords from list of dictionaries outputted by MLM and retain only sub-word token (without ##, no punctuation).
    :param mlm_results_list: list of dictionaries outputted by MLM pipeline
    :param nb_keywords_per_tweet: the number of keywords to retain per MLM operation
    :return: a list of the selected keywords from MLM results
    """
    selected_keywords_list = list()
    punctuation_list = [i for i in string.punctuation]
    for rank_mlm_keyword in range(nb_keywords_per_tweet):
        keyword = mlm_results_list[rank_mlm_keyword]['token_str']
        keyword = keyword.replace('##', '')
        if keyword not in punctuation_list:
            selected_keywords_list.append(keyword)
    return selected_keywords_list


def mlm_with_given_keyword(df, keyword, model_name, nb_keywords_per_tweet):
    """
    Perform MLM on all tweets in df and save list of resulting keywords in a new column top_mlm_keywords.
    :param df: pandas DataFrame containing only tweets that contain keyword
    :param keyword: high-lift keyword of interest which will be replaced by a [MASK] token for MLM
    :param model_name: name of BERT-based model used for MLM
    :param nb_keywords_per_tweet: number of tweets to retain from top MLM results
    :return: the df from the input with an extra column top_mlm_keywords containing the top nb_keywords_per_tweets MLM results in list format
    """
    mlm_pipeline = pipeline('fill-mask', model=model_name, tokenizer=model_name,
                            config=model_name, topk=nb_keywords_per_tweet)
    df['top_mlm_keywords'] = np.nan
    for tweet_index in range(df.shape[0]):
        tweet = df['text'][tweet_index]
        if tweet.count(keyword) > 1:
            n = random.randint(1, tweet.count(keyword))
            tweet = tweet.replace(keyword, '[MASK]', n).replace('[MASK]', keyword, n - 1)
        else:
            tweet = tweet.replace(keyword, '[MASK]')
        try:
            mlm_results_list = mlm_pipeline(tweet)
            df['top_mlm_keywords'][tweet_index] = extract_keywords_from_mlm_results(mlm_results_list,
                                                                                    nb_keywords_per_tweet=nb_keywords_per_tweet)
        except ValueError:
            print(f'Tweet giving ValueError during MLM pipeline: {tweet}')
            print(f'Keyword is {keyword}')
            df['top_mlm_keywords'][tweet_index] = list()
    return df


def mlm_with_selected_keywords(top_df, model_name, keyword_list, nb_tweets_per_keyword, nb_keywords_per_tweet,
                               lowercase):
    """
    For each keyword K in the keyword_list list, select nb_tweets_per_keyword tweets containing the keyword.
    For each of the nb_tweets_per_keyword tweets, do masked language on keyword K.
    :param top_df: pandas DataFrame containing top tweets (based on the based rate estimate)
    :param model_name: the BERT-based model from the Hugging Face model hub to use for MLM (complete list of names can be found here: https://huggingface.co/models
    :param keyword_list: the high-lift keywords to do MLM on
    :param nb_tweets_per_keyword: the number of tweets to retain for MLM per input keyword
    :param nb_keywords_per_tweet: the number of keywords to retain for each tweet used to do MLM
    :param lowercase: whether to lowercase input keywords
    :return: a dataframe containing all of the tweets containing at least one of the top-lift keywords in keyword_list
    with a top_lift_keyword column indicating which top-lift keyword is contained in the tweet and a top_mlm_keywords
    column containing a list of nb_keywords_per_tweet keywords outputted from MLM.
    """
    if lowercase:
        keyword_list = [keyword.lower() for keyword in keyword_list]
    # create empty dataframe and then concat with column top_lift_keyword and column MLM_keywords (list)
    # bootstrap in separate function
    tweets_all_top_lift_keywords_df = None
    for keyword in keyword_list:
        if tweets_all_top_lift_keywords_df is None:
            tweets_all_top_lift_keywords_df = sample_tweets_containing_selected_keywords(keyword=keyword,
                                                                                         nb_tweets_per_keyword=nb_tweets_per_keyword,
                                                                                         data_df=top_df,
                                                                                         lowercase=lowercase,
                                                                                         random=False)
            tweets_all_top_lift_keywords_df['top_lift_keyword'] = keyword
            tweets_all_top_lift_keywords_df = mlm_with_given_keyword(df=tweets_all_top_lift_keywords_df,
                                                                     keyword=keyword,
                                                                     model_name=model_name,
                                                                     nb_keywords_per_tweet=nb_keywords_per_tweet)
        else:
            tweets_containing_keyword_df = sample_tweets_containing_selected_keywords(keyword,
                                                                                      nb_tweets_per_keyword,
                                                                                      top_df, lowercase,
                                                                                      random=False)
            tweets_containing_keyword_df['top_lift_keyword'] = keyword
            tweets_containing_keyword_df = mlm_with_given_keyword(df=tweets_containing_keyword_df, keyword=keyword,
                                                                  model_name=model_name,
                                                                  nb_keywords_per_tweet=nb_keywords_per_tweet)
            tweets_all_top_lift_keywords_df = pd.concat([tweets_all_top_lift_keywords_df, tweets_containing_keyword_df])
    return tweets_all_top_lift_keywords_df


def bootstrapping(df, nb_samples, method):
    """
    If method is each_keyword_separately:
    For each top-lift keyword K, out of the X tweets containing K, produce nb_samples random samples of size X (sampling with replacement).
    For each random sample, collect the keywords resulting from MLM and append them to a list L.

    If method is all_keywords_together:
    Out of the X tweets containing at least one top-lift keyword, produce nb_samples random samples of size X (sampling with replacement).
    For each random sample, collect the keywords resulting from MLM and append them to a list L.

    :param df: a pandas DataFrame containing all tweets containing at least one top lift keyword (output of former function)
    :param nb_samples: number of bootstrapped samples
    :param method: whether to bootstrap the MLM results of all keywords together or do it separately for each kw
    :return: a dictionary containing the word count from list L (keys are words and values are word counts)
    """
    if method == "each_keyword_separately":
        top_lift_keywords_list = df.top_lift_keyword.unique()
        final_results_dict = dict()
        for keyword in top_lift_keywords_list:
            tweets_containing_keyword_df = df.loc[df['top_lift_keyword'] == keyword].reset_index(drop=True)
            all_top_mlm_keywords_list = list()
            for sample_nb in range(nb_samples):
                tweets_containing_keyword_sample_df = tweets_containing_keyword_df.sample(
                    n=tweets_containing_keyword_df.shape[0],
                    replace=True)
                all_top_mlm_keywords_list += tweets_containing_keyword_sample_df['top_mlm_keywords'].sum()
            final_results_dict[keyword] = all_top_mlm_keywords_list
        final_results_list = [item for sublist in list(final_results_dict.values()) for item in sublist]
        return dict(Counter(final_results_list))
    elif method == "all_keywords_together":
        all_top_mlm_keywords_list = list()
        for sample_nb in range(nb_samples):
            tweets_sample_df = df.sample(n=df.shape[0], replace=True)
            all_top_mlm_keywords_list += tweets_sample_df['top_mlm_keywords'].sum()
        return dict(Counter(all_top_mlm_keywords_list))


def eliminate_keywords_contained_in_positives_from_training(keyword_list, column):
    """
    Identify keywords in keyword_list contained in tweets labelled as positive for the training set of a given class.
    Delete these keywords from the input keyword_list
    :param keyword_list: List of keywords to look for in positives from the training set
    :param column: Class (e.g. lost_job_1mo, is_hired_1mo)
    :return: the inputted keyword_list without, if any, the keywords found in positives from the training set.
    """
    train_df = pd.read_csv(
        os.path.join('/scratch/mt4493/twitter_labor/twitter-labor-data/data/jul23_iter0/preprocessed',
                     'train_{}.csv'.format(column)),
        lineterminator='\n')
    positive_train_df = train_df[train_df['class'] == 1].reset_index(drop=True)
    final_keyword_list = list()
    for keyword in keyword_list:
        if positive_train_df['text'].str.contains(keyword).sum() == 0:
            final_keyword_list.append(keyword)
    return final_keyword_list


if __name__ == "__main__":
    # Define args from command line
    args = get_args_from_command_line()
    inference_folder_name = os.path.basename(os.path.dirname(args.inference_output_folder))
    # Calculate base rates
    labels = ['is_hired_1mo', 'is_unemployed', 'job_offer', 'job_search', 'lost_job_1mo']
    base_rates = [
        1.7342911457049017e-05,
        0.0003534645020523677,
        0.005604641971672389,
        0.00015839552996469054,
        1.455338466552472e-05]
    N_random = 92114009
    base_ranks = [int(x * N_random) for x in base_rates]
    label2rank = dict(zip(labels, base_ranks))
    for column in labels:
        print(f'************{column}************')
        top_tweets_folder_path = os.path.join(os.path.dirname(args.inference_output_folder), 'joined', column,
                                              f"top_tweets_{column}")
        top_tweets_filename = [file for file in os.listdir(top_tweets_folder_path) if file.endswith('.parquet')][0]
        # Load top tweets
        top_tweets_df = pd.read_parquet(os.path.join(top_tweets_folder_path, top_tweets_filename))
        print("Loaded top tweets")
        top_lift_keywords_list, keywords_with_lift_higher_1_list, full_random_wordcount_df = calculate_lift(
            top_df=top_tweets_df, nb_top_lift_kw=args.nb_top_lift_kw)
        print("Calculated lift")
        print(keywords_with_lift_higher_1_list)

        tweets_all_top_lift_keywords_df = mlm_with_selected_keywords(top_df=top_tweets_df,
                                                                     model_name='bert-base-cased',
                                                                     keyword_list=top_lift_keywords_list,
                                                                     nb_tweets_per_keyword=args.nb_tweets_per_kw_mlm,
                                                                     nb_keywords_per_tweet=args.nb_kw_per_tweet_mlm,
                                                                     lowercase=False
                                                                     )
        print("Ran MLM")
        keyword_count_dict = bootstrapping(df=tweets_all_top_lift_keywords_df, nb_samples=args.nb_bootstrapped_samples,
                                           method=args.bootstrapping_method)
        print("Performed bootstrapping")
        # keep only words with lift strictly higher than 1 and for which wordcount/total_nb_of_tweets > 1/100K
        full_random_wordcount_df['frequency'] = full_random_wordcount_df['count'] / N_random
        full_random_wordcount_df = full_random_wordcount_df.loc[
            full_random_wordcount_df['frequency'] > 1 / 100000].reset_index(drop=True)
        high_frequency_keywords_list = list(full_random_wordcount_df['word'].values)
        relevant_keywords_list = keywords_with_lift_higher_1_list + high_frequency_keywords_list
        relevant_keywords_list = [k for k in relevant_keywords_list if k in keyword_count_dict.keys()]
        keyword_count_dict = {k: keyword_count_dict[k] for k in list(set(relevant_keywords_list))}
        # keep top tweets in terms of wordcount in the overall output of MLM
        top_keyword_dict = Counter(keyword_count_dict).most_common(args.nb_final_candidate_kw)
        print("Final results before dropping keywords appearing in positives from training set:")
        print(top_keyword_dict)

        # diversity constraint (iteration 0): discard keywords contained in positives from training set
        current_keyword_list = eliminate_keywords_contained_in_positives_from_training(
            keyword_list=list(keyword_count_dict.keys()),
            column=column)
        keyword_count_dict = {k: keyword_count_dict[k] for k in current_keyword_list}
        # keep top tweets in terms of wordcount in the overall output of MLM
        top_keyword_dict = Counter(keyword_count_dict).most_common(args.nb_final_candidate_kw)
        print("Final results after dropping keywords appearing in positives from training set:")
        print(top_keyword_dict)

        # TO DO: Save whole results as parquet
        # final_keyword_count_df = pd.DataFrame(keyword_count_dict.items(), columns=['word', 'wordcount'])
        # final_keyword_count_df = final_keyword_count_df.sort_values(by=["wordcount"], ascending=False).reset_index(drop=True)
        # final_keyword_count_path = os.path.join(args.final_keywords_folder_path, )
        # final_keyword_count_df.to_parquet()

        # TO DO: drop tweets that are already labelled
