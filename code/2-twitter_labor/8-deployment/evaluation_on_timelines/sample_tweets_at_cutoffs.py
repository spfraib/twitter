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

if __name__ == '__main__':
    args = get_args_from_command_line()
    logger.info(args.country_code)
    # Define paths
    path_to_tweets = os.path.join(
        '/scratch/spf248/twitter/data/user_timeline/user_timeline_text_preprocessed', args.country_code)  # Random set of tweets
    model_folder_dict = {'US': 'iter_8-2000-3GB-9032948', 'BR': 'iter_9-2000-3GB-8406630'}
    path_to_scores = os.path.join('/scratch/spf248/twitter/data/labor/tweet_scores/BERT',
                                  args.country_code, model_folder_dict[args.country_code], 'output')
    path_to_evals = os.path.join(
        '/scratch/spf248/twitter/data/user_timeline/user_timeline_evaluation_samples', args.country_code)  # Where to store the sampled tweets to be labeled
    if not os.path.exists(path_to_evals):
        os.makedirs(path_to_evals)
    scores_df = pd.concat([pd.read_parquet(path) for path in Path(path_to_scores).glob('*.parquet')])
    scores_df = scores_df.reset_index()
    logger.info('Loaded scores')
    cutoff_df = pd.read_csv('/scratch/spf248/twitter/data/active_learning/evaluation_metrics/cutoffs.csv')
    cutoff_df = cutoff_df.loc[cutoff_df['country']==args.country_code].reset_index(drop=True)
    for label in ['is_hired_1mo', 'lost_job_1mo', 'is_unemployed', 'job_search', 'job_offer']:
        final_df_list = list()
        logger.info(f'Starting with {label}')
        # scores_df = scores_df.sort_values(by=[label], ascending=False)
        cutoff = cutoff_df.loc[cutoff_df['class']==label]['cutoff'].reset_index(drop=True)['cutoff'][0]
        scores_df['modified_score'] = scores_df[label] - cutoff
        above_threshold_df = scores_df.loc[scores_df['modified_score'] > 0].nsmallest(500, 'modified_score')
        below_threshold_df = scores_df.loc[scores_df['modified_score'] < 0].nlargest(500, 'modified_score')
        if above_threshold_df.shape[0] > 400:
            df = pd.concat([above_threshold_df, below_threshold_df]).sample(100)
            output_path = os.path.join(path_to_evals, f'ids_to_retrieve_{label}.parquet')
            scores_df.to_parquet(output_path, index=False)