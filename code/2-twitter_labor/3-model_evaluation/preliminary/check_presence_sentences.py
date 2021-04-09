import pandas as pd
import argparse
import logging
from pathlib import Path
import os
import re

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str,
                        default="US")
    parser.add_argument("--threshold", type=float,
                        default=0.95)
    args = parser.parse_args()
    return args

def sentence_in_string(sentence_list, mystring):
    if any(regex in mystring for regex in sentence_list):
        return 1
    else:
        return 0

if __name__ == '__main__':
    args = get_args_from_command_line()
    # define paths
    data_path = '/scratch/mt4493/twitter_labor/twitter-labor-data/data'
    random_path = f'{data_path}/random_samples/random_samples_splitted'
    random_path_evaluation = Path(os.path.join(random_path, args.country_code, 'evaluation'))
    # load random set
    random_df = pd.concat(
        pd.read_parquet(parquet_file)
        for parquet_file in random_path_evaluation.glob('*.parquet')
    )
    logger.info('Loaded random data')
    random_df['text_lower'] = random_df['text'].str.lower()
    labels = ['is_hired_1mo', 'lost_job_1mo', 'is_unemployed', 'job_offer', 'job_search']
    sentences_path = '/scratch/mt4493/twitter_labor/twitter-labor-data/data/evaluation_metrics/US/recall/scored_sentences'
    for label in labels:
        print(label)
        sentences_df = pd.read_csv(os.path.join(sentences_path, f'US-{label}.csv'))
        sentences_list = sentences_df['text'].tolist()
        sentences_list = [sentence.lower() for sentence in sentences_list]
        random_df[f'sentence_{label}'] = random_df['text_lower'].apply(lambda x: sentence_in_string(sentence_list=sentences_list, mystring=x))
        # print(random_df[f'sentence_{label}'].value_counts(dropna=False))
        # print(random_df[f'sentence_{label}'].value_counts(dropna=False, normalize=True))
    random_df = random_df[['tweet_id'] + [f'sentence_{label}' for label in labels]]
    output_path = f'/scratch/mt4493/twitter_labor/twitter-labor-data/data/random_samples/random_samples_splitted/{args.country_code}/evaluation_seedlist_keyword'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    random_df.to_parquet(os.path.join(output_path, 'evaluation_sentences.parquet'), index=False)