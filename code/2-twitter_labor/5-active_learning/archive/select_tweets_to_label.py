import os
import logging
import argparse
import pandas as pd
import numpy as np
import pytz
import pyarrow
from pathlib import Path
from transformers import BertTokenizer, BertModel, BertConfig, BertForTokenClassification, pipeline, \
    AutoModelForTokenClassification, AutoTokenizer
import torch
import nltk
from nltk.corpus import stopwords
import string
from nltk.util import skipgrams
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from collections import Counter

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# TODO although we decided on starting values for the parameters, it's unclear what the correct values should be. Nir
#  suggests that we start with these parameters and save the output of each stage of this script to look at whether
#  the outputs we are getting make sense. e.g. :
#  - look at the values of final_selected_keywords_list to see if the keywords are meaninful
#  - look at the skip gram results to see if they are not too obvious/degenerate
#  - look at the nb_tweets_exploit top tweets to see if they really will help with training, etc




def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_output_folder", type=str,
                        help="Path to the inference folder containing the merged parquet file.")
    parser.add_argument("--tweets_to_label_output_path", type=str,
                        help="Path to the folder where the tweets to label are saved.")
    parser.add_argument("--train_data_folder", type=str,
                        help="Path of folder when train and val data are stored.")
    parser.add_argument("--exploit_method", type=str,
                        help="Method chosen for the exploit part", default='top_tweets')
    parser.add_argument("--nb_tweets_exploit", type=int,
                        help="Number of tweets from exploit part to send to labelling (Exploit).")
    parser.add_argument("--nb_top_lift_kw", type=int,
                        help="Number of top-lift keywords to keep for MLM (Keyword exploration).")
    parser.add_argument("--nb_tweets_per_kw_mlm", type=int,
                        help="Number of tweets to use per keyword to do MLM (Keyword exploration).",
                        default=1)
    parser.add_argument("--nb_kw_per_tweet_mlm", type=int,
                        help="Number of keywords to extract from MLM for each tweet (Keyword exploration). ")
    parser.add_argument("--nb_tweets_per_kw_final", type=int,
                        help="Number of tweets to send to labelling per keyword outputted by MLM (Keyword exploration).")
    parser.add_argument("--nb_top_kskipngrams", type=int,
                        help="Number of top-lift k-skip-n-grams to keep (Sentence exploration). ")
    parser.add_argument("--nb_tweets_per_kskipngrams", type=int,
                        help="Number of tweets to send to labelling per k-skip-n-gram (Sentence exploration). ")
    parser.add_argument("--k_skipgram", type=int, help="k from k-skip-n-gram (Sentence exploration).", default=2)
    parser.add_argument("--n_skipgram", type=int, help="n from k-skip-n-gram (Sentence exploration).", default=3)
    args = parser.parse_args()
    return args


def get_token_in_sequence_with_most_attention(model, tokenizer, input_sequence):
    """
    Run an input sequence through the BERT model, collect and average attention scores per token and return token with
    most average attention.
    """
    tokenized_input_sequence = tokenizer.tokenize(input_sequence)
    input_ids = torch.tensor(tokenizer.encode(input_sequence, add_special_tokens=False)).unsqueeze(0)
    outputs = model(input_ids)
    last_hidden_states, pooler_outputs, hidden_states, attentions = outputs
    attention_tensor = torch.squeeze(torch.stack(attentions))
    attention_tensor_averaged = torch.mean(attention_tensor, (0, 1))
    attention_average_scores_per_token = torch.sum(attention_tensor_averaged, dim=0)
    attention_scores_dict = dict()
    for token_position in range(len(tokenized_input_sequence)):
        attention_scores_dict[token_position] = attention_average_scores_per_token[token_position].item()
    print(attention_scores_dict)
    return {'token_index': max(attention_scores_dict, key=attention_scores_dict.get),
            'token_str': tokenized_input_sequence[max(attention_scores_dict, key=attention_scores_dict.get)]}


def drop_stopwords_punctuation(df):
    """
    Drop rows containing stopwords or punctuation in the word column of the input dataframe.
    :param df: input pandas DataFrame, must contain a word column
    :return: pandas DataFrame without rows containing stopwords or punctuation
    """
    punctuation_list = [i for i in string.punctuation]
    all_stops = stopwords.words('english') + punctuation_list
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
    wordcount_df = top_wordcount_df.join(full_random_wordcount_df, on=['word'])
    wordcount_df = drop_stopwords_punctuation(wordcount_df)
    wordcount_df['lift'] = (wordcount_df['count_top_tweets'] / wordcount_df[
        'count']) * N_random / label2rank[column]
    wordcount_df = wordcount_df.sort_values(by=["lift"], ascending=False).reset_index()
    # Keep only word with lift > 1
    wordcount_df = wordcount_df[wordcount_df['lift'] > 1]
    if wordcount_df.shape[0] < nb_top_lift_kw:
        return wordcount_df['word'].tolist()
    else:
        return wordcount_df['word'][:nb_top_lift_kw].tolist()


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
    if tweets_containing_keyword_df.shape[0] < nb_tweets_per_keyword:
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


def mlm_with_selected_keywords(top_df, model_name, keyword_list, nb_tweets_per_keyword, nb_keywords_per_tweet,
                               lowercase):
    """
    For each keyword K in the keyword_list list, select nb_tweets_per_keyword tweets containing the keyword.
    For each of the nb_tweets_per_keyword tweets, do masked language on keyword K.
    Retain the top nb_keywords_per_tweet keywords from MLM and store them in the final_selected_keywords list.
    :param top_df: pandas DataFrame containing top tweets (based on the based rate estimate)
    :param model_name: the BERT-based model from the Hugging Face model hub to use for MLM (complete list of names can be found here: https://huggingface.co/models
    :param keyword_list: the high-lift keywords to do MLM on
    :param nb_tweets_per_keyword: the number of tweets to retain for MLM per input keyword
    :param nb_keywords_per_tweet: the number of keywords to retain for each tweet used to do MLM
    :param lowercase: whether to lowercase input keywords
    :return: a list of keywords combining all results from MLM
    """

    # run mask on all tweets but then do bootstrap
    # bootstrap:
    #     take random sample with replacement of top tweets
    #     take top 10 words that come out of the mask
    # take the top 10 by global boostrap count
    # do the 1/100K threshold like before

    # pick the top 10-20 most frequent in the 100M
    # TODO check the freq of the keywords in final_selected_keywords_list and make sure that they are more than
    #  1/100K and that the keywords have lift > 1 (otherwise they are not important). freq here is defined as %
    #  of tweets where keywords where it appears

    # TODO number 2: do this over bootstraps from the random sample to make sure we are not overfitting


    mlm_pipeline = pipeline('fill-mask', model=model_name, tokenizer=model_name,
                            config=model_name, topk=nb_keywords_per_tweet)
    final_selected_keywords_list = list()
    if lowercase:
        keyword_list = [keyword.lower() for keyword in keyword_list]
    for keyword in keyword_list:
        tweets_containing_keyword_df = sample_tweets_containing_selected_keywords(keyword, nb_tweets_per_keyword,
                                                                                  top_df, lowercase, random=False)
        for tweet_index in range(tweets_containing_keyword_df.shape[0]):
            tweet = tweets_containing_keyword_df['text'][tweet_index]
            tweet = tweet.replace(keyword, '[MASK]')
            mlm_results_list = mlm_pipeline(tweet)
            # TODO check that extract_keywords_from_mlm_results works correctly?
            final_selected_keywords_list = + extract_keywords_from_mlm_results(mlm_results_list,
                                                                               nb_keywords_per_tweet=nb_keywords_per_tweet)
    return final_selected_keywords_list


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


def k_skip_n_grams(sent, k, n):
    """
    Apply the skipgrams method from NLTK and return the results in list format.
    :param sent: input sentence in which to look for k-skip-n-grams
    :param k: k parameter from k-skip-n-grams
    :param n: n parameter from k-skip-n-grams
    :return: a list containing all the k-skip-n-grams found in sent
    """
    return list(skipgrams(sent, k=k, n=n))


def drop_tweet_if_already_labelled(data_df, column, train_data_folder):
    """
    For each column, drop tweets that were already labelled from the random set used for active learning.
    :param data_df: pandas DataFrame containing the random set
    :param column: class
    :param train_data_folder: folder where the train/val data is stored
    :return: pandas DataFrame containing the random set without the tweets that were already labelled
    """
    train_df = pd.read_csv(os.path.join(train_data_folder, 'train_{}.csv'.format(column)), lineterminator='\n')
    train_df = train_df.set_index('tweet_id')
    val_df = pd.read_csv(os.path.join(train_data_folder, 'val_{}.csv'.format(column)), lineterminator='\n')
    val_df = val_df.set_index('tweet_id')
    already_labelled_index = train_df.index.append(val_df.index)
    data_df = data_df.set_index('tweet_id')
    data_df = data_df.drop(already_labelled_index)
    return data_df.reset_index()


def exploit_part(all_data_df, top_df, method, nb_tweets_exploit, column):
    """
    Select tweets to label as part of the exploit part, given the chosen selection method.
    :param all_data_df: pandas DataFrame containing the whole random set
    :param top_df: pandas DataFrame containing the top tweets (based on the base rate calculation)
    :param method: the version of the exploitation process.
    - 'random_sample_top_tweets': take a random sample of nb_tweets_exploit tweets in the top_df
    - 'sample_base_rank': take nb_tweets_exploit tweets around the base rank
    - 'top_tweets': take nb_tweets_exploit tweets starting from the top of the distribution
    :param nb_tweets_exploit: number of tweets to label for the exploit part
    :param column: class
    :return: pandas DataFrame containing tweets to label (tweet_id and text)
    """
    # TODO: if all the tweets selecting are positive (high positive probability, expand until we start seeing a big
    #  decay. Do it manually first to see if this is even a problem (look at top nb_tweets_exploit tweet's scores
    #  and see if any are low enough

    # TODO number 2: do this over bootstraps from the random sample to make sure we are not overfitting

    if method == "random_sample_top_tweets":
        exploit_data_df = top_df.sample(n=nb_tweets_exploit).reset_index(drop=True)
    elif method == "sample_base_rank":
        base_rank = label2rank[column]
        interval = int(nb_tweets_exploit / 2)
        exploit_data_df = all_data_df[base_rank - interval:base_rank + interval]
    elif method == "top_tweets":
        exploit_data_df = all_data_df[:nb_tweets_exploit]
    exploit_data_df = exploit_data_df[['tweet_id', 'text']]
    exploit_data_df['source'] = 'exploit'
    exploit_data_df['label'] = column
    return exploit_data_df


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
        # Load data, compute skipgrams
        print('loading tweets...', "{}_all.parquet".format(column))
        # input_parquet_path = os.path.join(args.inference_output_folder, column, "{}_all.parquet".format(column))
        input_parquet_path = os.path.join(args.inference_output_folder, column, "sample_{}_all.parquet".format(column))
        #DEBUG
        all_data_df = pd.read_parquet(input_parquet_path)
        print('loaded', all_data_df.shape)
        all_data_df['skipgrams'] = all_data_df['tokenized_preprocessed_text'].apply(k_skip_n_grams, k=args.k_skipgram,
                                                                                    n=args.n_skipgram)
        all_data_df = drop_tweet_if_already_labelled(data_df=all_data_df, column=column,
                                                     train_data_folder=args.train_data_folder)

        # TODO: is this data already sorted by column? otherwise this is not selecting the top tweets
        top_df = all_data_df[:label2rank[column]]
        # EXPLOITATION (final data in exploit_data_df)
        exploit_data_df = exploit_part(all_data_df=all_data_df, top_df=top_df, method=args.exploit_method,
                                       nb_tweets_exploit=args.nb_tweets_exploit, column=column)
        # KEYWORD EXPLORATION (w/ keyword lift; final data in explore_kw_data_df)
        ## identify top lift keywords
        explore_kw_data_df = top_df
        top_lift_keywords_list = calculate_lift(explore_kw_data_df, nb_top_lift_kw=args.nb_top_lift_kw)
        ## for each top lift keyword X, identify Y top tweets containing X and do MLM
        selected_keywords_list = mlm_with_selected_keywords(top_df=explore_kw_data_df, model_name='bert-base-cased',
                                                            keyword_list=top_lift_keywords_list,
                                                            nb_tweets_per_keyword=args.nb_kw_per_tweet_mlm,
                                                            nb_keywords_per_tweet=5*args.nb_kw_per_tweet_mlm,
                                                            lowercase=True
                                                            )
        ## diversity constraint (iteration 0)
        final_selected_keywords_list = eliminate_keywords_contained_in_positives_from_training(selected_keywords_list,
                                                                                               column)
        ## select nb_kw_per_tweet_mlm keywords from list of final keywords
        final_selected_keywords_list = final_selected_keywords_list[:args.nb_kw_per_tweet_mlm]
        ## select final tweets
        tweets_to_label = exploit_data_df
        for final_keyword in final_selected_keywords_list:
            sample_tweets_containing_final_keyword_df = sample_tweets_containing_selected_keywords(
                keyword=final_keyword,
                nb_tweets_per_keyword=args.nb_tweets_per_kw_final,
                data_df=all_data_df, lowercase=True, random=True)
            sample_tweets_containing_final_keyword_df = sample_tweets_containing_final_keyword_df[['tweet_id', 'text']]
            sample_tweets_containing_final_keyword_df['label'] = column
            sample_tweets_containing_final_keyword_df['keyword'] = keyword
            sample_tweets_containing_final_keyword_df['source'] = 'explore_keyword'
            tweets_to_label = tweets_to_label.append(sample_tweets_containing_final_keyword_df, ignore_index=True)

        # SENTENCE EXPLORATION
        explore_st_data_df = top_df
        skipgrams_count = explore_st_data_df.explode('skipgrams').reset_index(drop=True)
        # Drop k-skip-n-grams for which 2/3 or more tokens are special characters (special tokens from preprocessing such as <hashtag> or punctuation)
        # TODO Dhaval has not looked through this section very carefully yet (hard to do without even sample data)!
        # TODO number 2: unclear what the correct n and k should be. TBD by playing with the data
        skipgrams_count['share_specific_tokens'] = skipgrams_count['skipgrams'].apply(
            lambda token_list: sum('<' in token for token in [str(i) for i in token_list]) / len(token_list))
        skipgrams_count['share_punctuation'] = skipgrams_count['skipgrams'].apply(
            lambda token_list: len(list(set(token_list).intersection(punctuation_list))) / len(token_list))
        skipgrams_count['total_share_irrelevant_tokens'] = skipgrams_count['share_specific_tokens'] + skipgrams_count[
            'share_punctuation']
        skipgrams_count = skipgrams_count[skipgrams_count['total_share_irrelevant_tokens'] < (2 / 3)].reset_index(
            drop=True)
        # Store top k-skip-n-grams in a list
        top_structures_dict = dict(skipgrams_count['skipgrams'].value_counts(dropna=False))
        top_structures_list = [item[0] for item in Counter(top_structures_dict).most_common(args.nb_top_kskipngrams)]
        # For each top structure, identify tweets containing structure, sample and store in tweets_to_label dataframe
        for top_structure in top_structures_list:
            all_data_df["{}_in_skipgrams".format(top_structure)] = [single_top_structure in single_structure_list for
                                                                    single_top_structure, single_structure_list in
                                                                    zip([top_structure] * all_data_df.shape[0],
                                                                        all_data_df['skipgrams'])]
            sample_tweets_containing_final_structure_df = all_data_df[
                all_data_df["{}_in_skipgrams".format(top_structure)]].reset_index(drop=True)
            sample_tweets_containing_final_structure_df = sample_tweets_containing_final_structure_df.sample(
                n=args.nb_tweets_per_kskipngrams).reset_index(drop=True)
            sample_tweets_containing_final_structure_df = sample_tweets_containing_final_structure_df[
                ['tweet_id', 'text']]
            sample_tweets_containing_final_structure_df['label'] = column
            sample_tweets_containing_final_structure_df['k-skip-n-gram'] = top_structure
            sample_tweets_containing_final_structure_df['source'] = 'explore_sentence'
            tweets_to_label = tweets_to_label.append(sample_tweets_containing_final_structure_df, ignore_index=True)

        # Save tweets to label
        output_folder_path = os.path.join(tweets_to_label_output_path, inference_folder_name)
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
        output_file_path = os.path.join(output_folder_path, '{}_to_label.csv'.format(column))
        tweets_to_label.to_csv(output_file_path)

        break #DEBUG for just one column