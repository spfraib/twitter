import os
import numpy as np
import pandas as pd
import argparse
import pyarrow
import subprocess
from pathlib import Path


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str)
    parser.add_argument("--model_folder", type=str)
    args = parser.parse_args()
    return args


def get_sampled_indices(n_sample=10, n_cutoff=5):
    sampled_points = []  # index of scores around which we sample n_sample tweets
    sampled_ranks = []  # ranks of sampled tweets
    for point, rank in enumerate(sorted(set([int(x) for i in range(n_cutoff) for x in np.logspace(i, i + 1, i + 1)]))):
        if not point:
            new_ranks = list(range(rank, rank + n_sample))
        else:
            new_ranks = list(range(rank + 1, rank + n_sample + 1))
        print('Index of sampled point:', point)
        print('Sampled ranks:', new_ranks)
        sampled_points.extend([point] * n_sample)
        sampled_ranks.extend(new_ranks)
    return sampled_points, sampled_ranks


if __name__ == '__main__':
    args = get_args_from_command_line()
    for label in ['is_hired_1mo', 'lost_job_1mo', 'is_unemployed', 'job_search', 'job_offer']:
        path_to_tweets = Path(
            os.path.join('/scratch/mt4493/twitter_labor/twitter-labor-data/data/random_samples/random_samples_splitted',
                         args.country_code,
                         'evaluation'))  # Random set of tweets
        path_to_scores = Path(
            os.path.join('/scratch/mt4493/twitter_labor/twitter-labor-data/data/inference', args.country_code,
                         args.model_folder, 'output',
                         label))  # Prediction scores from classification
        path_to_evals = os.path.join('/scratch/mt4493/twitter_labor/twitter-labor-data/data/evaluation_inference',
                                     args.country_code, args.model_folder,
                                     label)  # Where to store the sampled tweets to be labeled
        if not os.path.exists(path_to_evals):
            os.makedirs(path_to_evals)
        sampled_points, sampled_ranks = get_sampled_indices()
        print('# Sampled points:', len(set(sampled_points)))
        print('# Sampled tweets:', len(sampled_ranks))
        tweets_df = pd.concat(pd.read_parquet(parquet_file) for parquet_file in path_to_tweets.glob('*.parquet'))
        scores_df = pd.concat(pd.read_parquet(parquet_file) for parquet_file in path_to_scores.glob('*.parquet'))
        sampled_indices = pd.DataFrame({
            'point': sampled_points,
            'rank': sampled_ranks})
        df = tweets_df.merge(scores_df, on=['tweet_id'])
        df['rank'] = df['score'].rank(method='dense', ascending=False)
        df = df.sort_values(by=['rank'], ascending=False).reset_index(drop=True)
        df = df.merge(sampled_indices, on=['rank'])
        df = df[['rank', 'tweet_id', 'text', 'score', 'point']]
        df.to_csv(path_to_evals, index=False)
