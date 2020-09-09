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

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_output_folder", type=str,
                        help="Path to the inference data. Must be in csv format.",
                        default="")
    parser.add_argument("--N_exploit", type=int, help="Number of tweets to label for exploitation.", default="")
    parser.add_argument("--N_explore_kw", type=int, help="Number of tweets to label for keyword explore.",
                        default="")
    parser.add_argument("--K_tw_exploit", type=int, help="Number of top tweets to study for exploitation.", default="")
    parser.add_argument("--K_tw_explore_kw", type=int, help="Number of top tweets to study for keyword explore.",
                        default="")
    parser.add_argument("--K_tw_explore_sent", type=int, help="Number of top tweets to study for sentence explore.",
                        default="")
    parser.add_argument("--K_kw_explore", type=int,
                        help="Number of keywords to select for keyword explore.", default="")

    args = parser.parse_args()
    return args


def get_token_in_sequence_with_most_attention(model, tokenizer, input_sequence):
    """Run an input sequence through the BERT model, collect and average attention scores per token and return token with most average attention."""
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


def extract_keywords_from_mlm_results(mlm_results_list, K_kw_explore):
    selected_keywords_list = list()
    for rank_mlm_keyword in range(K_kw_explore):
        selected_keywords_list.append(mlm_results_list[rank_mlm_keyword]['token_str'])
    return selected_keywords_list


def drop_stopwords_punctuation(df):
    punctuation_list = [i for i in string.punctuation]
    all_stops = stopwords.words('english') + punctuation_list
    df = df[~df['word'].isin(all_stops)].reset_index(drop=True)
    return df


def calculate_lift(top_df, nb_keywords):
    top_wordcount_df = top_df.explode('tokenized_text')
    top_wordcount_df = top_wordcount_df['tokenized_text'].value_counts().rename_axis(
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
    if wordcount_df.shape[0] < nb_keywords:
        return wordcount_df['word'].tolist()
    else:
        return wordcount_df['word'][:nb_keywords].tolist()


def sample_tweets_containing_selected_keywords(keyword, nb_tweets_per_keyword, data_df, lowercase):
    if not lowercase:
        tweets_containing_keyword_df = data_df[data_df['text'].str.contains(keyword)].reset_index(drop=True)
    else:
        tweets_containing_keyword_df = data_df[data_df['lowercased_text'].str.contains(keyword)].reset_index(drop=True)
    tweets_containing_keyword_df = tweets_containing_keyword_df.sort_values(by=["score"], ascending=False).reset_index(
        drop=True)
    if tweets_containing_keyword_df.shape[0] < nb_tweets_per_keyword:
        print("Only {} tweets containing keyword {} (< {}). Sending all of them to labelling.".format(
            str(tweets_containing_keyword_df.shape[0]), keyword, str(nb_tweets_per_keyword)))
        return tweets_containing_keyword_df
    else:
        return tweets_containing_keyword_df[:nb_tweets_per_keyword]


def mlm_with_selected_keywords(top_df, model_name, keyword_list, nb_tweets_per_keyword, nb_keywords_per_tweet,
                               lowercase):
    """
    For each keyword K in the keyword_list list, select nb_tweets_per_keyword tweets containing the keyword.
    For each of the nb_tweets_per_keyword tweets, do masked language on keyword K.
    Retain the top nb_keywords_per_tweet keywords from MLM and store them in the final_selected_keywords list.
    """
    mlm_pipeline = pipeline('fill-mask', model=model_name, tokenizer=model_name,
                            config=model_name, topk=nb_keywords_per_tweet)
    final_selected_keywords_list = list()
    if lowercase:
        keyword_list = [keyword.lower() for keyword in keyword_list]
    for keyword in keyword_list:
        tweets_containing_keyword_df = sample_tweets_containing_selected_keywords(keyword, nb_tweets_per_keyword,
                                                                                  top_df, lowercase)
        for tweet_index in range(tweets_containing_keyword_df.shape[0]):
            tweet = tweets_containing_keyword_df['text'][tweet_index]
            tweet = tweet.replace(keyword, '[MASK]')
            mlm_results_list = mlm_pipeline(tweet)
            final_selected_keywords_list = + extract_keywords_from_mlm_results(mlm_results_list, nb_keywords_per_tweet)
    return final_selected_keywords_list


def eliminate_keywords_contained_in_positives_from_training(keyword_list, column):
    train_df = pd.read_csv(
        os.path.join('/scratch/mt4493/twitter_labor/twitter-labor-data/data/jul23_iter0/preprocessed',
                     'train_{}.csv'.format(column)),
        lineterminator='\n')
    positive_train_df = train_df[train_df['class']==1].reset_index(drop=True)
    final_keyword_list = list()
    for keyword in keyword_list:
        if positive_train_df['text'].str.contains(keyword).sum() == 0:
            final_keyword_list.append(keyword)
    return final_keyword_list




if __name__ == "__main__":
    # Define args from command line
    args = get_args_from_command_line()
    mlm_pipeline = pipeline('fill-mask', model='bert-base-uncased', tokenizer='bert-base-uncased',
                            config='bert-base-uncased', topk=10)
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
        # Load model, config and tokenizer
        config = BertConfig.from_pretrained(PATH_MODEL_FOLDER, output_hidden_states=True, output_attentions=True)
        tokenizer = BertTokenizer.from_pretrained(PATH_MODEL_FOLDER)
        model = BertModel.from_pretrained(PATH_MODEL_FOLDER, config=config)
        # Load data, drop RT and tokenize text
        input_parquet_path = os.path.join(args.inference_output_folder, column, "{}_all.parquet".format(column))
        all_data_df = pd.read_parquet(input_parquet_path)
        all_data_df = all_data_df[~all_data_df.text.str.contains("RT", na=False)].reset_index(drop=True)
        all_data_df = all_data_df.sort_values(by=["score"], ascending=False).reset_index(drop=True)
        all_data_df['tokenized_text'] = all_data_df['text'].apply(tokenizer.tokenize)
        all_data_df['lowercased_text'] = all_data_df['text'].str.lower()
        # EXPLOITATION (final data in exploit_data_df)
        exploit_data_df = all_data_df[:args.N_exploit]
        # KEYWORD EXPLORATION (w/ keyword lift; final data in explore_kw_data_df)
        ## identify top lift keywords
        nb_tweets_per_keyword = int(args.N_explore_kw / args.K_kw_explore)
        explore_kw_data_df = all_data_df[:label2rank[column]]
        top_lift_keywords_list = calculate_lift(explore_kw_data_df, nb_keywords=10)
        ## for each top lift keyword X, identify Y top tweets containing X and do MLM
        selected_keywords_list = mlm_with_selected_keywords(top_df=explore_kw_data_df, model_name='bert-base-cased',
                                                             keyword_list=top_lift_keywords_list,
                                                             nb_tweets_per_keyword=nb_tweets_per_keyword,
                                                             nb_keywords_per_tweet=1, lowercase=True
                                                             )
        ## diversity constraint (iteration 0)
        final_selected_keywords_list = eliminate_keywords_contained_in_positives_from_training(selected_keywords_list, column)
        

        explore_kw_data_df = sample_tweets_containing_selected_keywords(selected_keywords_list, nb_tweets_per_keyword,
                                                                        explore_kw_data_df)

        # explore (attention version)
        # explore_kw_data_df = all_data_df[:args.K_tw_explore_kw]
        # nb_tweets_per_keyword = args.N_explore_kw // (args.K_tw_explore_kw * args.K_kw_explore)
        # for tweet_rank in range(explore_kw_data_df.shape[0]):
        #    tweet_str = explore_kw_data_df['text'][tweet_rank]
        #    tokenized_tweet = tokenizer.tokenize(tweet)
        #    # Determine the token with the highest average attention and replace it with a [MASK] token
        #    attention_token_index = \
        #    get_token_in_sequence_with_most_attention(model=model, tokenizer=tokenizer, input_sequence=tokenized_tweet)[
        #        'token_index']
        #    tokenized_tweet[attention_token_index] = '[MASK]'
        #    # Do MLM and select the top K_kw_explore keywords
        #    mlm_results_list = mlm_pipeline(' '.join(tokenized_tweet))
        #    selected_keywords_list = extract_keywords_from_mlm_results(mlm_results_list, args.K_kw_explore)
        #    # For each keyword W, draw a random sample of nb_tweets_per_keyword tweets containing W
        #    for selected_keyword in selected_keywords_list:
        ## Don't forget to lower
