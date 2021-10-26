import os
import numpy as np
import pandas as pd
import argparse
import subprocess
from pathlib import Path
from itertools import product
import logging

logging.basicConfig(
                    # filename=f'{args.log_path}.log',
                    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str)
    args = parser.parse_args()
    return args


def get_sampled_indices(min_magnitude=3, top_tweets_threshold=1000000000):
    index_list = list(range(1,11)) + list(range(91,101))
    start=991
    range_value = int((top_tweets_threshold-9-start)/pow(10,min_magnitude)) + 1
    print(range_value)
    for i in range(range_value):
        list_to_add = list(range(start+pow(10,min_magnitude)*i, start+pow(10,min_magnitude)*i + 10))
        index_list += list_to_add
    return index_list

if __name__ == '__main__':
    args = get_args_from_command_line()
    logger.info(args.country_code)
    # Load sample indices
    index_list = get_sampled_indices()
    # Define paths
    path_to_tweets = os.path.join(
        '/scratch/spf248/twitter/data/user_timeline/user_timeline_text_preprocessed', args.country_code)  # Random set of tweets
    model_folder_dict = {'US': 'iter_8-2000-3GB-9032948', 'BR': 'iter_9-2000-3GB-8406630'}
    path_to_scores = os.path.join('/scratch/spf248/twitter/data/labor/tweet_scores/BERT',
                                  args.country_code, model_folder_dict[args.country_code], 'output')
    path_to_evals = os.path.join(
        '/scratch/spf248/twitter/data/user_timeline', 'user_timeline_evaluation_samples', args.country_code)  # Where to store the sampled tweets to be labeled
    if not os.path.exists(path_to_evals):
        os.makedirs(path_to_evals)
    for label in ['is_hired_1mo', 'lost_job_1mo', 'is_unemployed', 'job_search', 'job_offer']:
        final_df_list = list()
        logger.info(f'Starting with {label}')
        scores_df = pd.concat([pd.read_parquet(path) for path in list(Path(path_to_scores).glob('*.parquet'))[:1]])
        logger.info('Loaded scores')
        logger.info(f'Scores df columns: {list(scores_df.columns)}')
        scores_df['rank'] = scores_df[label].rank(method='first', ascending=False)
        scores_df = scores_df.loc[scores_df['rank'].isin(index_list)].reset_index(drop=True)
        logger.info('Selected indices. Now retrieving tweets with indices')
        for path in Path(path_to_tweets).glob('*.parquet'):
            tweets_df = pd.read_parquet(path, columns=['tweet_id', 'text'])
            logger.info(path)
            if 'tweet_id' in tweets_df.columns:
                tweets_df = tweets_df.loc[tweets_df['tweet_id'].isin(list(scores_df['tweet_id'].unique()))]
                if tweets_df.shape[0] > 0:
                    final_df_list.append(tweets_df)
            else:
                logger.info(f'No tweet_id column for {path}')
        logger.info('Finished retrieving tweets with indices.')
        tweets_df = pd.concat(final_df_list).reset_index(drop=True)
        df = tweets_df.merge(scores_df, on=['tweet_id']).reset_index(drop=True)
        df = df[['tweet_id', 'text', label, 'rank']]
        df.columns = ['tweet_id', 'text', 'score', 'rank']
        df = df.sort_values(by=['score'], ascending=False).reset_index(drop=True)
        output_path = os.path.join(path_to_evals, f'{label}.parquet')
        # df.to_parquet(output_path, index=False)
