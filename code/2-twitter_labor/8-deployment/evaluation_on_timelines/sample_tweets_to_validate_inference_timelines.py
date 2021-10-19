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
    parser.add_argument("--model_folder", type=str)
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
    # Load tweets
    path_to_tweets = os.path.join(
        '/scratch/spf248/twitter/data/user_timeline/user_timeline_text_preprocessed', args.country_code)  # Random set of tweets
    tweets = pd.concat([pd.read_parquet(path) for path in Path(path_to_tweets).glob('*.parquet')])
    logger.info('Loaded tweets')
    tweets = tweets[['tweet_id', 'text']]
    tweets = tweets.set_index('tweet_id')
    model_folder_dict = {'US': 'iter_8-2000-3GB-9032948', 'BR': 'iter_9-2000-3GB-8406630'}
    path_to_scores = os.path.join('/scratch/spf248/twitter/data/labor/tweet_scores/BERT',
                                  args.country_code, model_folder_dict[args.country_code], 'output')
    scores = pd.concat([pd.read_parquet(path) for path in Path(path_to_tweets).glob('*.parquet')])
    logger.info('Loaded scores')
    # Load sample indices
    index_list = get_sampled_indices()
    path_to_evals = os.path.join(
        '/scratch/spf248/twitter/data/user_timeline', 'evaluation_sample', args.country_code)  # Where to store the sampled tweets to be labeled
    if not os.path.exists(path_to_evals):
        os.makedirs(path_to_evals)
    for label in ['is_hired_1mo', 'lost_job_1mo', 'is_unemployed', 'job_search', 'job_offer']:
        logger.info(f'Starting with {label}')
        label_df = scores.sort_values(by=[label]).reset_index(drop=True)
        label_df['rank'] = label_df[label].rank(method='first', ascending=False)
        label_df = label_df.loc[label_df['rank'].isin(index_list)]
        final_df = tweets.merge(label_df, on=['tweet_id']).reset_index(drop=True)
        final_df = final_df[['tweet_id', 'text', label, 'rank']]
        final_df.columns = ['tweet_id', 'text', 'score', 'rank']
        output_path = os.path.join(path_to_evals, f'{label}.parquet')
        df.to_parquet(output_path, index=False)
