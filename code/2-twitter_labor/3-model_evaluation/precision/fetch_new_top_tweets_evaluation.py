import os
import numpy as np
import pandas as pd
import argparse
import subprocess
from pathlib import Path
from itertools import product
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str, default='US')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args_from_command_line()
    # Load tweets
    path_to_tweets = os.path.join(
        '/scratch/mt4493/twitter_labor/twitter-labor-data/data/random_samples/random_samples_splitted',
        args.country_code,
        'evaluation')  # Random set of tweets
    tweets = pd.concat([pd.read_parquet(path) for path in Path(path_to_tweets).glob('*.parquet')])
    tweets = tweets[['tweet_id', 'text']]

    # model_folder_list = ['iter_0-convbert-969622-evaluation', 'iter_1-convbert-3050798-evaluation',
    #                      'iter_2-convbert-3134867-evaluation',
    #                      'iter_3-convbert-3174249-evaluation', 'iter_4-convbert-3297962-evaluation',
    #                      'iter_0-convbert-969622-evaluation',
    #                      'iter_1-convbert_adaptive-5612019-evaluation', 'iter_2-convbert_adaptive-5972342-evaluation',
    #                      'iter_3-convbert_adaptive-5998181-evaluation', 'iter_4-convbert_adaptive-6057405-evaluation']
    # model_folder_list = ['iter_1-convbert_uncertainty-6200469-evaluation',
    #                      'iter_2-convbert_uncertainty-6253253-evaluation',
    #                      'iter_3-convbert_uncertainty-6318280-evaluation',
    #                      ]
    # model_folder_list = ['iter_4-convbert_uncertainty-6423646-evaluation']
    # model_folder_list = ['iter_1-convbert_uncertainty_uncalibrated-6480837-evaluation',
    #                      'iter_2-convbert_uncertainty_uncalibrated-6578026-evaluation',
    #                      'iter_3-convbert_uncertainty_uncalibrated-6596620-evaluation']
    # model_folder_list = ['iter_4-convbert_uncertainty_uncalibrated-6653849-evaluation']
    model_folder_dict = {'MX': ['iter_0-beto-3201262-evaluation', 'iter_1-beto-3741011-evaluation',
                         'iter_2-beto-4141605-evaluation',
                         'iter_3-beto-4379208-evaluation', 'iter_4-beto-4608158-evaluation'],
                        'BR': ['iter_0-bertimbau-2877651-evaluation', 'iter_1-bertimbau-3774133-evaluation',
                               'iter_2-bertimbau-4180985-evaluation', 'iter_3-bertimbau-4518774-evaluation',
                               'iter_4-bertimbau-4688729-evaluation']}
    model_folder_list = model_folder_dict[args.country_code]
    for model_folder in model_folder_list:
        logger.info(f'Folder: {model_folder}')
        path_to_evals = os.path.join(
            '/scratch/mt4493/twitter_labor/twitter-labor-data/data/active_learning/evaluation_inference',
            args.country_code, model_folder)  # Where to store the sampled tweets to be labeled
        if not os.path.exists(path_to_evals):
            os.makedirs(path_to_evals)
        for label in ['is_hired_1mo', 'lost_job_1mo', 'is_unemployed', 'job_search', 'job_offer']:
            logger.info(f'Class: {label}')
            path_to_scores = os.path.join('/scratch/mt4493/twitter_labor/twitter-labor-data/data/inference',
                                          args.country_code, model_folder, 'output',
                                          label)  # Prediction scores from classification
            scores = pd.concat([pd.read_parquet(path) for path in Path(path_to_scores).glob('*.parquet')]).reset_index()
            scores['rank'] = scores['score'].rank(method='first', ascending=False)
            scores = scores[scores['rank'].between(21, 50)]
            df = tweets.merge(scores, on=['tweet_id'])
            df = df.sort_values(by=['rank'], ascending=True).reset_index(drop=True)
            output_path = os.path.join(path_to_evals, f'extra_{label}.csv')
            df.to_csv(output_path, index=False)
