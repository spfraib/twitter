import pandas as pd
import argparse
import logging
from pathlib import Path
import os
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import re
import itertools
import json
import torch

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str)
    parser.add_argument("--selection_method", type=str, default='threshold_calibrated')
    parser.add_argument("--threshold", type=float,
                        default=0.95)
    parser.add_argument("--topk", type=int, default=1000000)
    args = parser.parse_args()
    return args


def compute_mean_max_diversity(matrix):
    max_list = list()
    for i in range(matrix.size()[0]):
        tensor = matrix[i] * -1
        max_list.append(tensor.max().item())
    return np.average(max_list)


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
    # Load env vars
    SLURM_ARRAY_TASK_ID = get_env_var('SLURM_ARRAY_TASK_ID', 0)
    SLURM_ARRAY_TASK_COUNT = get_env_var('SLURM_ARRAY_TASK_COUNT', 1)
    SLURM_JOB_ID = get_env_var('SLURM_JOB_ID', 1)

    # Define and select combination
    labels = ['job_search', 'job_offer', 'is_hired_1mo', 'lost_job_1mo', 'is_unemployed']
    combinations_list = list(itertools.product(
        *[['US'], labels]))
    selected_combinations = list(np.array_split(
        combinations_list,
        SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID])
    # selected_combinations = [('exploit_explore_retrieval', 5, 'lost_job_1mo')]
    logger.info(f'Selected combinations: {selected_combinations}')

    # results_dict = dict()
    # rank_dict = {
    #     'is_hired_1mo': 50000,
    #     'is_unemployed': 50000,
    #     'job_offer': 800000,
    #     'job_search': 250000,
    #     'lost_job_1mo': 10000}
    if len(selected_combinations) == 1:
        combination = selected_combinations[0]
        country_code = combination[0]
        label = combination[1]
        # load model
        diversity_model_dict = {
            'US': 'stsb-roberta-large',
            'MX': 'distiluse-base-multilingual-cased-v2',
            'BR': 'distiluse-base-multilingual-cased-v2'}
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        diversity_model = SentenceTransformer(diversity_model_dict[country_code], device=device)

        # if embeddings don't exist, load text
        embeddings_folder= f'/scratch/mt4493/twitter_labor/twitter-labor-data/data/keyword_model/sample_diversity/{country_code}/embeddings'
        if not os.path.exists(embeddings_folder):
            os.makedirs(embeddings_folder)

        embeddings_path = os.path.join(embeddings_folder, f'embeddings_keyword_{label}.pt')
        if not os.path.exists(embeddings_path):
            # load random set
            path_parquet = f'/scratch/mt4493/twitter_labor/twitter-labor-data/data/keyword_model/sample_diversity/{country_code}/{label}.parquet'
            random_df = pd.read_parquet(path_parquet)
            logger.info('Loaded data')
            tweet_list = random_df['text'].tolist()

            # compute and save diversity score
            embeddings = diversity_model.encode(tweet_list, convert_to_tensor=True)
            logger.info('Converted tweets to embeddings')
            torch.save(embeddings, embeddings_path)
            logger.info(f'Saved at {embeddings_path}')
        else:
            logger.info('Embeddings already exist')
