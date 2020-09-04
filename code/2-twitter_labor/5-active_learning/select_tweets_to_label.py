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
    parser.add_argument("--N_explore_kw", type=int, help="Number of tweets to label for keyword exploration.",
                        default="")
    parser.add_argument("--K_tw_exploit", type=int, help="Number of top tweets to study for exploitation.", default="")
    parser.add_argument("--K_tw_explore_kw", type=int, help="Number of top tweets to study for keyword exploration.",
                        default="")
    parser.add_argument("--K_tw_explore_sent", type=int, help="Number of top tweets to study for sentence exploration.",
                        default="")
    parser.add_argument("--K_kw_explore", type=int,
                        help="Number of keywords to select per top tweet for keyword exploration.", default="")

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
        selected_keywords.append(mlm_results_list[rank_mlm_keyword]['token_str'])
    return selected_keywords_list


if __name__ == "__main__":
    # Define args from command line
    args = get_args_from_command_line()
    mlm_pipeline = pipeline('fill-mask', model='bert-base-uncased', tokenizer='bert-base-uncased',
                            config='bert-base-uncased', topk=10)
    for column in ['is_hired_1mo', 'is_unemployed', 'job_offer', 'job_search', 'lost_job_1mo']:
        # Load model, config and tokenizer
        config = BertConfig.from_pretrained(PATH_MODEL_FOLDER, output_hidden_states=True, output_attentions=True)
        tokenizer = BertTokenizer.from_pretrained(PATH_MODEL_FOLDER)
        model = BertModel.from_pretrained(PATH_MODEL_FOLDER, config=config)
        # Load data
        input_parquet_path = os.path.join(args.inference_output_folder, column, "{}_all.parquet".format(column))
        all_data_df = pd.read_parquet(input_parquet_path)
        all_data_df = all_data_df.sort_values("score", inplace=True).reset_index()
        # exploit
        exploit_data_df = all_data_df[:args.N_exploit]
        # exploration (attention version)
        exploration_kw_data_df = all_data_df[:args.K_tw_explore_kw]
        nb_tweets_per_keyword = args.N_explore_kw // (args.K_tw_explore_kw * args.K_kw_explore)
        for tweet_rank in range(exploration_kw_data_df.shape[0]):
            tweet_str = exploration_kw_data_df['text'][tweet_rank]
            tokenized_tweet = tokenizer.tokenize(tweet)
            # Determine the token with the highest average attention and replace it with a [MASK] token
            attention_token_index = \
            get_token_in_sequence_with_most_attention(model=model, tokenizer=tokenizer, input_sequence=tokenize_tweet)[
                'token_index']
            tokenized_tweet[attention_token_index] = '[MASK]'
            # Do MLM and select the top K_kw_explore keywords
            mlm_results_list = mlm_pipeline(' '.join(tokenized_tweet))
            selected_keywords_list = extract_keywords_from_mlm_results(mlm_results_list, args.K_kw_explore)
            # For each keyword W, draw a random sample of nb_tweets_per_keyword tweets containing W
            for selected_keyword in selected_keywords_list:
        ## Don't forget to lower
