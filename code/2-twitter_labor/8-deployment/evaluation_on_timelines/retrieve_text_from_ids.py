from pathlib import Path
import argparse
import pandas as pd
import logging
from glob import glob
import os
import numpy as np

logging.basicConfig(
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

def get_env_var(varname, default):
    if os.environ.get(varname) != None:
        var = int(os.environ.get(varname))
        print(varname, ':', var)
    else:
        var = default
        print(varname, ':', var, '(Default)')
    return var

if __name__ == '__main__':
    args = get_args_from_command_line()
    logger.info(args.country_code)
    # define paths
    path_to_tweets = os.path.join(
        '/scratch/spf248/twitter/data/user_timeline/user_timeline_text_preprocessed', args.country_code)
    path_to_samples= os.path.join(
        '/scratch/spf248/twitter/data/user_timeline/user_timeline_evaluation_samples', args.country_code)
    SLURM_ARRAY_TASK_ID = get_env_var('SLURM_ARRAY_TASK_ID', 0)
    SLURM_ARRAY_TASK_COUNT = get_env_var('SLURM_ARRAY_TASK_COUNT', 1)
    SLURM_JOB_ID = get_env_var('SLURM_JOB_ID', 1)
    input_files_list = glob(os.path.join(path_to_tweets, '*.parquet'))
    paths_to_random = list(np.array_split(
        input_files_list,
        SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID])
    # load ids to retrieve
    for path in paths_to_random:
        tweets_df = pd.read_parquet(path)
        for label in  ['is_hired_1mo', 'lost_job_1mo', 'is_unemployed', 'job_search', 'job_offer']:
            id_path = os.path.join(path_to_samples, label, 'tweet_ids', 'ids_to_retrieve.parquet')
            id_df = pd.read_parquet(id_path)
            tweet_sample_df = tweets_df.loc[tweets_df['tweet_id'].isin(id_df['tweet_id'].tolist())]
            if tweet_sample_df.shape[0] > 0:
                output_path = os.path.join(path_to_samples, label, 'tweets')
                if not os.path.exists(output_path):
                    os.makedirs(output_path, exist_ok=True)
                tweet_sample_df.to_parquet(os.path.join(output_path, os.path.basename(path)), index=False)

    # for path in Path(path_to_tweets).glob('*.parquet'):
    #     tweets_df = pd.read_parquet(path, columns=['tweet_id', 'text'])
    #     logger.info(path)
    #     tweets_df = tweets_df.loc[tweets_df['tweet_id'].isin(list(scores_df['tweet_id'].unique()))]
    #     if tweets_df.shape[0] > 0:
    #         final_df_list.append(tweets_df)
    # logger.info('Finished retrieving tweets with indices.')
    # tweets_df = pd.concat(final_df_list).reset_index(drop=True)
    # df = tweets_df.merge(scores_df, on=['tweet_id']).reset_index(drop=True)
    # df = df[['tweet_id', 'text', label, 'rank']]
    # df.columns = ['tweet_id', 'text', 'score', 'rank']
    # df = df.sort_values(by=['score'], ascending=False).reset_index(drop=True)
    # output_path = os.path.join(path_to_evals, f'{label}.parquet')
    # df.to_parquet(output_path, index=False)