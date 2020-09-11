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


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_output_folder", type=str,
                        help="Path to the inference folder containing the parquet files.",
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
    positive_train_df = train_df[train_df['class'] == 1].reset_index(drop=True)
    final_keyword_list = list()
    for keyword in keyword_list:
        if positive_train_df['text'].str.contains(keyword).sum() == 0:
            final_keyword_list.append(keyword)
    return final_keyword_list


def k_skip_n_grams(sent, k, n):
    return list(skipgrams(sent, k=k, n=n))


text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
               'time', 'url', 'date', 'number'],
    # terms that will be annotated
    annotate={"hashtag", "allcaps", "elongated", "repeated",
              'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens

    # corpus from which the word statistics are going to be used
    # for word segmentation
    segmenter="twitter",

    # corpus from which the word statistics are going to be used
    # for spell correction
    corrector="twitter",

    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words

    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,

    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)


def ekphrasis_preprocessing(tweet):
    return " ".join(text_processor.pre_process_doc(tweet))


if __name__ == "__main__":
    # Define args from command line
    args = get_args_from_command_line()
    # Define MLM pipline
    mlm_pipeline = pipeline('fill-mask', model='bert-base-uncased', tokenizer='bert-base-uncased',
                            config='bert-base-uncased', topk=10)
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
        # Load model, config and tokenizer
        tokenizer = BertTokenizer.from_pretrained('DeepPavlov/bert-base-cased-conversational')
        # Load data, compute skipgrams
        input_parquet_path = os.path.join(args.inference_output_folder, column, "{}_all_sorted.parquet".format(column))
        all_data_df = pd.read_parquet(input_parquet_path)
        all_data_df['skipgrams'] = all_data_df['tokenized_preprocessed_text'].apply(k_skip_n_grams, k=TO_BE_DEFINED,
                                                                           n=TO_BE_DEFINED)
        top_df = all_data_df[:label2rank[column]]
        # EXPLOITATION (final data in exploit_data_df)
        exploit_data_df = all_data_df[:args.N_exploit]
        exploit_data_df = exploit_data_df[['tweet_id','text']]
        exploit_data_df['source'] = 'exploit'
        exploit_data_df['label'] = column
        # KEYWORD EXPLORATION (w/ keyword lift; final data in explore_kw_data_df)
        ## identify top lift keywords
        final_nb_tweets_per_keyword = TO_BE_DEFINED
        explore_kw_data_df = top_df
        top_lift_keywords_list = calculate_lift(explore_kw_data_df, nb_keywords=TO_BE_DEFINED)
        ## for each top lift keyword X, identify Y top tweets containing X and do MLM
        selected_keywords_list = mlm_with_selected_keywords(top_df=explore_kw_data_df, model_name='bert-base-cased',
                                                            keyword_list=top_lift_keywords_list,
                                                            nb_tweets_per_keyword=TO_BE_DEFINED,
                                                            nb_keywords_per_tweet=TO_BE_DEFINED, lowercase=True
                                                            )
        ## diversity constraint (iteration 0)
        final_selected_keywords_list = eliminate_keywords_contained_in_positives_from_training(selected_keywords_list,
                                                                                               column)
        ## select final tweets
        tweets_to_label = exploit_data_df
        for final_keyword in final_selected_keywords_list:
            sample_tweets_containing_final_keyword_df = sample_tweets_containing_selected_keywords(
                keyword=final_keyword,
                nb_tweets_per_keyword=final_nb_tweets_per_keyword,
                data_df=all_data_df, lowercase=True)
            sample_tweets_containing_final_keyword_df = sample_tweets_containing_final_keyword_df[['tweet_id','text']]
            sample_tweets_containing_final_keyword_df['label'] = column
            sample_tweets_containing_final_keyword_df['keyword'] = keyword
            sample_tweets_containing_final_keyword_df['source'] = 'explore_keyword'
            tweets_to_label = tweets_to_label.append(sample_tweets_containing_final_keyword_df, ignore_index=True)

        # SENTENCE EXPLORATION
        nb_top_structures = TO_BE_DEFINED
        explore_st_data_df = top_df
        skipgrams_count = explore_st_data_df.explode('skipgrams').reset_index(drop=True)
        # Drop k-skip-n-grams for which 2n/3 or more tokens are special characters (special tokens from preprocessing such as <hashtag> or punctuation)
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
        top_structures_list = [item[0] for item in Counter(top_structures_dict).most_common(TO_BE_DEFINED)]
        for top_structure in top_structures_list:
            all_data_df["{}_in_skipgrams".format(top_structure)] = [single_top_structure in single_structure_list for
                                                                    single_top_structure, single_structure_list in
                                                                    zip([top_structure] * all_data_df.shape[0], all_data_df['skipgrams'])]
            sample_tweets_containing_final_structure_df = all_data_df[all_data_df["{}_in_skipgrams".format(top_structure)]].reset_index(drop=True)
            sample_tweets_containing_final_structure_df['label'] = column
            sample_tweets_containing_final_structure_df['k-skip-n-gram'] = top_structure
            sample_tweets_containing_final_structure_df['source'] = 'explore_sentence'
            tweets_to_label = tweets_to_label.append(sample_tweets_containing_final_structure_df, ignore_index=True)
        

