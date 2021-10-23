import os
import numpy as np
import pandas as pd
import argparse
import subprocess
from pathlib import Path
from itertools import product


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
        scores_df = pd.concat([pd.read_parquet(path, columns=['tweet_id', label]) for path in Path(path_to_scores).glob('*.parquet')])
        logger.info('Loaded scores')
        scores_df = scores_df.sort_values(by=[label]).reset_index(drop=True)
        scores_df['rank'] = scores_df[label].rank(method='first', ascending=False)
        scores_df = scores_df.loc[scores_df['rank'].isin(index_list)].reset_index(drop=True)
        for path in Path(path_to_tweets).glob('*.parquet'):
            tweets_df = pd.read_parquet(path, columns=['tweet_id', 'text'])
            tweets_df = tweets_df.loc[tweets_df['tweet_id'].isin(list(scores_df['tweet_id'].unique()))]
            if tweets_df.shape[0] > 0:
                final_df = scores_df.merge(tweets_df, on=['tweet_id'])
                final_df_list.append(final_df)
        df = pd.concat(final_df_list).reset_index(drop=True)
        df = df[['tweet_id', 'text', label, 'rank']]
        df.columns = ['tweet_id', 'text', 'score', 'rank']
        output_path = os.path.join(path_to_evals, f'{label}.parquet')
        df.to_parquet(output_path, index=False)