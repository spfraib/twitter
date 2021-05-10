import torch
import numpy as np
from numba import jit
from time import time
import pandas as pd
import itertools
import os
import logging
import statistics
import math
import json
import argparse

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str,
                        default="US")
    parser.add_argument("--method", type=str)
    args = parser.parse_args()
    return args


@jit(nopython=True)
def pairwise_sim(E, D):
    D[0][0] = 1
    N = E.shape[0]
    for j in range(1, N):
        if j % 10000 == 0:
            print(j)
        D[j][j] = 1
        for i in range(j):
            d = np.dot(E[i], E[j]) / (np.linalg.norm(E[i]) * np.linalg.norm(E[j]))
            D[i][j] = d
            D[j][i] = d
    return D


@jit(nopython=True)
def sim(E1,E2,D):
    N1 = E1.shape[0]
    N2 = E2.shape[0]
    for i in range(N1):
        if i % 10000 == 0:
            print(i)
        for j in range(N2):
            D[i][j] = np.dot(E1[i],E2[j])/(np.linalg.norm(E1[i])*np.linalg.norm(E2[j]))
    return D

def get_env_var(varname, default):
    if os.environ.get(varname) != None:
        var = int(os.environ.get(varname))
        print(varname, ':', var)
    else:
        var = default
        print(varname, ':', var, '(Default)')
    return var


def mean_diversity(D):
    sim_list = 0
    nb_pairs = 0
    for i in range(D.shape[0]):
        for j in range(i):
            sim_list.append(D[i][j])
            nb_pairs += 1
    return {
        'mean_diversity': (-1 / nb_pairs) * sum(sim_list),
        'mean_diversity_std_error': statistics.stdev(sim_list) / math.sqrt(nb_pairs)}


def mean_max_diversity(D):
    max_dist_list = list()
    for i in range(D.shape[0]):
        dist_array = 1 - D[i]
        max_dist_list.append(dist_array.max())
    return {
        'mean_max_diversity': statistics.mean(max_dist_list),
        'mean_max_diversity_std_error': statistics.stdev(max_dist_list) / math.sqrt(len(max_dist_list))}


if __name__ == '__main__':
    args = get_args_from_command_line()

    path = 'embeddings_adaptive_iter4_lost_job_1mo.pt'
    # Load env vars
    SLURM_ARRAY_TASK_ID = get_env_var('SLURM_ARRAY_TASK_ID', 0)
    SLURM_ARRAY_TASK_COUNT = get_env_var('SLURM_ARRAY_TASK_COUNT', 1)
    SLURM_JOB_ID = get_env_var('SLURM_JOB_ID', 1)
    # Build combinations
    labels = ['job_search', 'job_offer', 'is_hired_1mo', 'lost_job_1mo', 'is_unemployed']
    combinations_list = list(itertools.product(*[['our_method', 'adaptive', 'uncertainty'], range(5), labels]))
    combinations_list = [combination for combination in combinations_list if
                         combination[:2] not in [('adaptive', 0), ('uncertainty', 0), ('uncertainty', 4)]]
    selected_combinations = list(np.array_split(
        combinations_list,
        SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID])
    logger.info(f'Selected combination(s): {selected_combinations}')
    if len(selected_combinations) == 1:
        # define main variables
        combination = selected_combinations[0]
        al_method = combination[0]
        iter_nb = int(combination[1])
        label = combination[2]
        # prepare results_dict
        results_dict = dict()
        results_dict[al_method] = dict()
        results_dict[al_method][iter_nb] = dict()
        # define output paths
        data_path = '/scratch/mt4493/twitter_labor/twitter-labor-data/data'
        output_path = f'{data_path}/evaluation_metrics/US/diversity/threshold_calibrated_{args.method}'
        output_file = os.path.join(output_path, f"diversity_{al_method}_{iter_nb}_{label}.json")
        if not os.path.exists(output_file):
            embeddings_path = f'{data_path}/evaluation_metrics/US/diversity/embeddings/embeddings_{al_method}_iter{iter_nb}_{label}.pt'
            embeddings = torch.load(embeddings_path, map_location=torch.device('cpu'))
            logger.info('Loaded embeddings')
            embeddings = embeddings.cpu().detach().numpy()
            logger.info(f'Embedding size: {embeddings.shape[0]}')
            if args.method == 'pairwise':
                # embeddings_orig = embeddings.copy()
                D = np.zeros((embeddings.shape[0], embeddings.shape[0]))
                logger.info('Starting to calculate pairwise similarity')
                D = pairwise_sim(E=embeddings, D=D)
                logger.info('Done. Computing diversity metrics')
                mean_diversity_dict = mean_diversity(D)
                mean_max_diversity_dict = mean_max_diversity(D)
                results_dict[al_method][iter_nb][label] =  {**mean_diversity_dict, **mean_max_diversity_dict}
            elif args.method == 'distance_with_seed':
                positive_seed_embedding_path = f'{data_path}/evaluation_metrics/US/diversity/embeddings_positive_train/jan5_iter0_{label}.pt'
                embeddings_positive_seed = torch.load(positive_seed_embedding_path, map_location=torch.device('cpu'))
                embeddings_positive_seed = embeddings_positive_seed.cpu().detach().numpy()
                logger.info(f'# of seed positives: {embeddings_positive_seed.shape[0]}')
                D = np.zeros((embeddings.shape[0], embeddings_positive_seed.shape[0]))
                logger.info('Starting to calculate pairwise similarity')
                D = sim(E1=embeddings, E2=embeddings_positive_seed, D=D)
                logger.info('Done. Computing diversity metrics')
                mean_max_diversity_dict = mean_max_diversity(D)
                results_dict[al_method][iter_nb][label] =  mean_max_diversity_dict
            logger.info(f'Metrics: {results_dict}')
            # save results
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            with open(output_file, 'w') as f:
                json.dump(results_dict, f)
            logger.info(f'Metrics saved at {output_file}')
        # E1 = embeddings_orig
        # E2 = np.tile(embeddings_orig,(2,1))
        # D = np.zeros((E1.shape[0],E2.shape[0]))
        # dist(E1,E2,D)

        # t0 = time()
        # timing = []
        # for i in np.arange(1,6):
        #     E = np.tile(embeddings_orig,(i,1))
        #     D = np.zeros((E.shape[0],E.shape[0]))
        #     pairwise_dist(E,D)
        #     t1 = time()
        #     timing.append((i,E.shape[0],t1-t0))
        #     print((i,E.shape[0],t1-t0))
        #     t0 = time()
        # timing = pd.DataFrame(timing,columns=['index','n_tweets','time_in_sec'])
