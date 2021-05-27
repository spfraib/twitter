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
from ranks import ranks_dict
import torch

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str,
                        default="US")
    parser.add_argument("--selection_method", type=str, default='threshold_calibrated')
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
            3: 'iter_3-convbert_uncertainty-6318280-evaluation',
            4: 'iter_4-convbert_uncertainty-6423646-evaluation'}}
    # Define and select combination
    labels = ['job_search', 'job_offer', 'is_hired_1mo', 'lost_job_1mo', 'is_unemployed']
    combinations_list = list(itertools.product(*[['our_method', 'adaptive', 'uncertainty'], range(5), labels]))
    combinations_list = [combination for combination in combinations_list if
                         combination[:2] not in [('adaptive', 0), ('uncertainty', 0)]]
    selected_combinations = list(np.array_split(
        combinations_list,
        SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID])
    logger.info(f'Selected combinations: {selected_combinations}')

    results_dict = dict()
    if len(selected_combinations) == 1:
        combination = selected_combinations[0]
        inference_folder = inference_folder_dict[combination[0]][int(combination[1])]
        al_method = combination[0]
        iter_nb = int(combination[1])
        label = combination[2]
        logger.info(f'Active learning method: {al_method}')
        logger.info(f'Iteration number: {iter_nb}')
        logger.info(f'Class: {label}')
        if al_method not in results_dict.keys():
            results_dict[al_method] = dict()
        if iter_nb not in results_dict[al_method].keys():
            results_dict[al_method][iter_nb] = dict()
        if label not in results_dict[al_method][iter_nb].keys():
            results_dict[al_method][iter_nb][label] = dict()

        # load model
        diversity_model_dict = {
            'US': 'stsb-roberta-large',
            'MX': 'distiluse-base-multilingual-cased-v2',
            'BR': 'distiluse-base-multilingual-cased-v2'}
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        diversity_model = SentenceTransformer(diversity_model_dict[args.country_code], device=device)

        # if embeddings don't exist, load text
        embeddings_path = f'/scratch/mt4493/twitter_labor/twitter-labor-data/data/evaluation_metrics/US/diversity/embeddings/embeddings_{al_method}_iter{iter_nb}_{label}.pt'
        if not os.path.exists(embeddings_path):
            # load random set
            random_df = pd.concat(
                pd.read_parquet(parquet_file)
                for parquet_file in random_path_new_samples.glob('*.parquet')
            )
            logger.info('Loaded random data')

            # load scores
            inference_path = os.path.join(data_path, 'inference')
            scores_path = Path(os.path.join(inference_path, args.country_code, inference_folder, 'output', label))
            scores_df = pd.concat(
                pd.read_parquet(parquet_file)
                for parquet_file in scores_path.glob('*.parquet')
            )
            logger.info('Loaded scores')
            all_df = scores_df.merge(random_df, on="tweet_id", how='inner')
            logger.info('Merged random set and scores')
            all_df = all_df.sort_values(by=['score'], ascending=False).reset_index(drop=True)

            if args.selection_method == 'threshold':
                all_df = all_df.loc[all_df['score'] > args.threshold].reset_index(drop=True)
                logger.info(f'# tweets with score > {args.threshold}: {all_df.shape[0]}')
            elif args.selection_method == 'topk':
                all_df = all_df[:args.topk]
            elif args.selection_method == 'threshold_calibrated':
                rank = ranks_dict[al_method][iter_nb][label]['numerator']
                all_df = all_df[:rank]
                logger.info(f'# tweets with calibrated score > 0.5: {all_df.shape[0]}')
            all_df['inference_folder'] = inference_folder
            tweet_list = all_df['text'].tolist()

        # compute and save diversity score
        embeddings_path = f'/scratch/mt4493/twitter_labor/twitter-labor-data/data/evaluation_metrics/US/diversity/embeddings/embeddings_{al_method}_iter{iter_nb}_{label}.pt'
        if os.path.exists(embeddings_path):
            embeddings = torch.load(embeddings_path)
            logger.info('Loaded embeddings')
            logger.info(f'Size: {embeddings.size()}')
        else:
            embeddings = diversity_model.encode(tweet_list, convert_to_tensor=True)
            logger.info('Converted tweets to embeddings')
            torch.save(embeddings, embeddings_path)
            logger.info(f'Saved at {embeddings_path}')
        cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)
        logger.info('Computed cosine similarity')
        mean_diversity_score = (-torch.sum(cosine_scores) / (list(embeddings.size())[0] ** 2)).item()
        mean_max_diversity_score = compute_mean_max_diversity(matrix=cosine_scores)
        logger.info(f'Mean diversity score: {mean_diversity_score}')
        logger.info(f'Mean max diversity score: {mean_max_diversity_score}')
        results_dict[al_method][iter_nb][label]['mean_diversity_score'] = mean_max_diversity_score
        results_dict[al_method][iter_nb][label]['mean_max_diversity_score'] = mean_max_diversity_score

    # save results
    if args.selection_method == 'threshold':
        folder_name = f'threshold_{int(args.threshold * 100)}'
    elif args.selection_method == 'topk':
        folder_name = f'top_{args.topk}'
    elif args.selection_method == 'threshold_calibrated':
        folder_name = f'threshold_calibrated'
    output_path = f'{data_path}/evaluation_metrics/{args.country_code}/diversity/{folder_name}'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # save results dict
    with open(os.path.join(output_path, f'diversity_{str(SLURM_ARRAY_TASK_ID)}.json'), 'w') as f:
        json.dump(results_dict, f)

