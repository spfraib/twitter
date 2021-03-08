import pandas as pd
from pathlib import Path
import numpy as np
import os
import argparse
import logging
from transformers import AutoTokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str,
                        default="US")
    parser.add_argument("--iter_number", type=str)
    parser.add_argument("--inference_folder", type=str)

    args = parser.parse_args()
    return args

def turn_token_array_to_string(x):
    list_x = list(x)
    return '_'.join(list_x)

tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/bert-base-cased-conversational')

def token_in_vocab(token, tokenizer):
    token_tokenized_list = tokenizer.convert_ids_to_tokens(tokenizer.encode(token))
    if '[UNK]' in token_tokenized_list:
        return False
    else:
        return True

def drop_ngram_oov(n, ngram_list, tokenizer):
    for ngram in ngram_list:
        for i in range(n):
            one_gram_str = ngram[i]
            one_gram_tokenized_list = tokenizer.convert_ids_to_tokens(tokenizer.encode(one_gram_str))



key2tokens = {'two_grams_is_hired_1mo': [('got', 'prom'),
  ('my', 'permit'),
  ('car', 'finally'),
  ('chapter', 'start'),
  ('paid', 'tomorrow'),
  ('started', 'workout'),
  ('early', 'yay'),
  ('license', 'today'),
  ('first', 'survived'),
  ('finished', 'homework')],
 'three_grams_is_hired_1mo': [('finally', 'got', 'phone'),
  ('!', 'my', 'woot'),
  ('first', 'tomorrow', 'work'),
  ('day', 'ready', 'start'),
  ('off', 'starting', 'year'),
  ('keys', 'new', 'the'),
  ('home', 'just', 'long'),
  ('friday', 'i', 'paid'),
  ('a', 'today', 'yay'),
  ('at', 'back', 'gym')],
 'two_grams_is_unemployed': [('im', 'losing'),
  ('lost', 'voice'),
  ('am', 'heartless'),
  ('have', 'migraine'),
  ('depressed', 'now'),
  ('food', 'starving'),
  ('mood', 'shitty'),
  ('attack', 'having'),
  ('hungover', 'still'),
  ('need', 'stressed')],
 'three_grams_is_unemployed': [('i', 'lost', 'voice'),
  ('am', 'losing', 'my'),
  ('have', 'headache', 'now'),
  ('a', 'breakdown', 'having'),
  ('eat', 'starving', 'to'),
  ('attack', 'had', 'just'),
  ('!', 'hungry', 'im'),
  ('been', 'for', 'sick'),
  ('bored', 'need', 'something'),
  ('depressed', 'do', 'not')],
 'two_grams_job_offer': [('bd', 'text'),
  ('buyer', 'looking'),
  ('becoming', 'interested'),
  ('ba', 'dm'),
  ('used', 'wd'),
  ('<money>', 'offering'),
  ('intern', 'internship'),
  ('auto', 'sale'),
  ('chevrolet', 'for'),
  ('local', 'wedding')],
 'three_grams_job_offer': [('bd', 'call', 'in'),
  ('ba', 'me', 'text'),
  ('estate', 'for', 'looking'),
  ('free', 'is', 'offering'),
  ('at', 'sale', 'wd'),
  ('buyer', 'on', 'real'),
  ('computer', 'deal', 'this'),
  ('auto', 'sales', 'used'),
  ('added', 'check', 'we'),
  ('!', '<email>', 'interested')],
 'two_grams_job_search': [('any', 'recommendations'),
  ('am', 'suggestions'),
  ('anyone', 'borrow'),
  ('hobby', 'need'),
  ('anybody', 'tonight'),
  ('i', 'takers'),
  ('ideas', 'something'),
  ('asap', 'some'),
  ('hmu', 'someone'),
  ('soon', 'trip')],
 'three_grams_job_search': [('any', 'i', 'recommendations'),
  ('?', 'am', 'suggestions'),
  ('find', 'need', 'something'),
  ('anyone', 'borrow', 'have'),
  ('buy', 'know', 'where'),
  ('soon', 'to', 'trip'),
  ('a', 'hobby', 'new'),
  ('give', 'me', 'ride'),
  ('back', 'get', 'gym'),
  ('good', 'in', 'places')],
 'two_grams_lost_job_1mo': [('just', 'kicked'),
  ('lost', 'power'),
  ('banned', 'got'),
  ('pulled', 'today'),
  ('bed', 'fell'),
  ('phone', 'yesterday'),
  ('flipped', 'off'),
  ('house', 'locked'),
  ('dumped', 'my'),
  ('blacked', 'i')],
 'three_grams_lost_job_1mo': [('got', 'just', 'kicked'),
  ('called', 'off', 'work'),
  ('lost', 'my', 'voice'),
  ('i', 'pulled', 'today'),
  ('house', 'locked', 'of'),
  ('for', 'hospital', 'out'),
  ('last', 'me', 'phone'),
  ('all', 'mad', 'over'),
  ('home', 'sent', 'to'),
  ('a', 'make', 'yesterday')]}

if __name__ == '__main__':
    args = get_args_from_command_line()
    output_path = f'/scratch/mt4493/twitter_labor/twitter-labor-data/data/active_learning/sampling_top_lift/{args.country_code}/{args.inference_folder}/labeling_sample'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for label in ['is_hired_1mo', 'is_unemployed', 'job_search', 'lost_job_1mo']:
        for ngram in ['two_grams', 'three_grams']:
            ngram_path = f'/scratch/mt4493/twitter_labor/twitter-labor-data/data/active_learning/sampling_top_lift/{args.country_code}/{args.inference_folder}/{label}/{ngram}'
            list_parquet = Path(ngram_path).glob('*.parquet')
            df = pd.concat(map(pd.read_parquet, list_parquet)).reset_index(drop=True)
            df['tokens_str'] = df['tokens'].apply(turn_token_array_to_string)
            df_sample = df.groupby('tokens_str', group_keys=False).apply(lambda df: df.sample(5)).reset_index(drop=True)
            df_sample = df_sample[['tweet_id', 'text', 'tokens_str']]
            df_sample.to_parquet(os.path.join(output_path, f'sample_{label}_{ngram}.parquet'), index=False)
            logging.info(f'Sample for label {label} {ngram} saved at: {output_path}/sample_{label}_{ngram}.parquet')

    for list_ngram in key2tokens.values():

    labeling_path = f'/scratch/mt4493/twitter_labor/twitter-labor-data/data/ngram_samples/{args.country_code}/iter{args.iter_number}/labeling'
    list_sample_parquet = Path(output_path).glob('*.parquet')
    df = pd.concat(map(pd.read_parquet, list_sample_parquet)).reset_index(drop=True)
    df.to_parquet(os.path.join(labeling_path, 'merged.parquet'))
    logging.info(f'Whole labeling set saved at {labeling_path}/merged.parquet')