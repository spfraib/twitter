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

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str,
                        default="US")
    parser.add_argument("--method", type=str, default='topk')
    parser.add_argument("--threshold", type=float,
                        default=0.95)
    parser.add_argument("--topk", type=int, default=10000)
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
    # define paths
    data_path = '/scratch/mt4493/twitter_labor/twitter-labor-data/data'
    random_path = f'{data_path}/random_samples/random_samples_splitted'
    random_path_new_samples = Path(os.path.join(random_path, args.country_code, 'evaluation'))

    inference_folder_dict = {
        'our_method': {
            0: 'iter_0-convbert-969622-evaluation',
            1: 'iter_1-convbert-3050798-evaluation',
            2: 'iter_2-convbert-3134867-evaluation',
            3: 'iter_3-convbert-3174249-evaluation',
            4: 'iter_4-convbert-3297962-evaluation'},
        'adaptive': {
            1: 'iter_1-convbert_adaptive-5612019-evaluation',
            2: 'iter_2-convbert_adaptive-5972342-evaluation',
            3: 'iter_3-convbert_adaptive-5998181-evaluation',
            4: 'iter_4-convbert_adaptive-6057405-evaluation'},
        'uncertainty': {
            1: 'iter_1-convbert_uncertainty-6200469-evaluation',
            2: 'iter_2-convbert_uncertainty-6253253-evaluation',
            3: 'iter_3-convbert_uncertainty-6318280-evaluation'}}
    # Define and select combination
    labels = ['job_search', 'job_offer', 'is_hired_1mo', 'lost_job_1mo', 'is_unemployed']
    combinations_list = list(itertools.product(*[['our_method', 'adaptive', 'uncertainty'], range(5), labels]))
    combinations_list = [combination for combination in combinations_list if
                         combination[:2] not in [('adaptive', 0), ('uncertainty', 0), ('uncertainty', 4)]]
    selected_combinations = list(np.array_split(
        combinations_list,
        SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID])
    logger.info(f'Selected combinations: {selected_combinations}')

    # load model
    diversity_model_dict = {
        'US': 'stsb-roberta-large',
        'MX': 'distiluse-base-multilingual-cased-v2',
        'BR': 'distiluse-base-multilingual-cased-v2'}
    diversity_model = SentenceTransformer(diversity_model_dict[args.country_code])

    # load random set
    random_df = pd.concat(
        pd.read_parquet(parquet_file)
        for parquet_file in random_path_new_samples.glob('*.parquet')
    )
    logger.info('Loaded random data')

    inference_path = os.path.join(data_path, 'inference')
    results_dict = dict()
    to_label_list = list()
    for combination in selected_combinations:
        inference_folder = inference_folder_dict[combination[0]][int(combination[1])]
        logger.info(f'**** Inference folder: {inference_folder} ****')
        results_dict[inference_folder] = dict()
        label = combination[2]
        logger.info(f'** Class: {label} **')
        scores_path = Path(os.path.join(inference_path, args.country_code, inference_folder, 'output', label))
        scores_df = pd.concat(
            pd.read_parquet(parquet_file)
            for parquet_file in scores_path.glob('*.parquet')
        )
        logger.info('Loaded scores')
        all_df = scores_df.merge(random_df, on="tweet_id", how='inner')
        all_df = all_df.sort_values(by=['score'], ascending=False).reset_index(drop=True)
        # # keep aside 50 top tweets
        # top_df = all_df[:50]
        # top_df['tweet_type'] = 'top_50'
        # restrict to score > T
        if args.method == 'threshold':
            all_df = all_df.loc[all_df['score'] > args.threshold].reset_index(drop=True)
            logger.info(f'# tweets with score > {args.threshold}: {all_df.shape[0]}')
        elif args.method == 'topk':
            all_df = all_df[:args.topk]
        all_df['inference_folder'] = inference_folder

        results_dict[inference_folder][label] = dict()
        # compute and save diversity score
        if all_df.shape[0] > 0:
            tweet_list = all_df['text'].tolist()
            embeddings = diversity_model.encode(tweet_list, convert_to_tensor=True)
            logger.info('Converted tweets to embeddings')
            cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)
            logger.info('Computed cosine similarity')
            diversity_score = compute_mean_max_diversity(matrix=cosine_scores)
            # diversity_score = (-torch.sum(cosine_scores) / (len(tweet_list) ** 2)).item()
            logger.info(f'Diversity score: {diversity_score}')
            results_dict[inference_folder][label]['diversity_score'] = diversity_score
        else:
            results_dict[inference_folder][label]['diversity_score'] = np.nan
    # save results
    if args.method == 'threshold':
        folder_name = f'threshold_{int(args.threshold * 100)}'
    elif args.method == 'topk':
        folder_name = f'top_{args.topk}'
    output_path = f'{data_path}/evaluation_metrics/{args.country_code}/diversity/{folder_name}'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # save results dict
    with open(os.path.join(output_path, f'diversity_{str(SLURM_ARRAY_TASK_ID)}.json'), 'w') as f:
        json.dump(results_dict, f)

