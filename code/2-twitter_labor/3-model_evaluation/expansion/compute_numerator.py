import pickle
import os
import numpy as np
import logging
from pathlib import Path
import pandas as pd
import argparse
from scipy import stats
from scipy import optimize
from statistics import stdev
import itertools
import json

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str, default='US')
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    return args

def func(x, param_pair, threshold):
    return 1 / (1 + np.exp(-(param_pair[0] * x + param_pair[1]))) - threshold

def func_mean(x, params, threshold):
    all_calibrated_scores = [1 / (1 + np.exp(-(param[0] * x + param[1]))) for param in params]
    return np.mean(all_calibrated_scores, axis=0) - threshold

def calibrate(score, param):
    return 1 / (1 + np.exp(-(param[0] * score + param[1])))

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
    SLURM_ARRAY_TASK_ID = get_env_var('SLURM_ARRAY_TASK_ID', 0)
    SLURM_ARRAY_TASK_COUNT = get_env_var('SLURM_ARRAY_TASK_COUNT', 1)
    SLURM_JOB_ID = get_env_var('SLURM_JOB_ID', 1)

    folder_dict = {
        'our_method': {
            0: [{
                'eval': 'iter_0-convbert-969622-evaluation',
                'new_samples': 'iter_0-convbert-1122153-new_samples'}, 'jan5_iter0'],
            1: [{
                'eval': 'iter_1-convbert-3050798-evaluation',
                'new_samples': 'iter_1-convbert-3062566-new_samples'}, 'feb22_iter1'],
            2: [{
                'eval': 'iter_2-convbert-3134867-evaluation',
                'new_samples': 'iter_2-convbert-3139138-new_samples'}, 'feb23_iter2'],
            3: [{
                'eval': 'iter_3-convbert-3174249-evaluation',
                'new_samples': 'iter_3-convbert-3178321-new_samples'}, 'feb25_iter3'],
            4: [{
                'eval': 'iter_4-convbert-3297962-evaluation',
                'new_samples': 'iter_4-convbert-3308838-new_samples'}, 'mar1_iter4']},
        'adaptive': {
            1: [{
                'eval': 'iter_1-convbert_adaptive-5612019-evaluation',
                'new_samples': 'iter_1-convbert_adaptive-5644714-new_samples'}, 'apr19_iter1_adaptive'],
            2: [{
                'eval': 'iter_2-convbert_adaptive-5972342-evaluation',
                'new_samples': 'iter_2-convbert_adaptive-5974832-new_samples'}, 'apr25_iter2_adaptive'],
            3: [{
                'eval': 'iter_3-convbert_adaptive-5998181-evaluation',
                'new_samples': 'iter_3-convbert_adaptive-6006450-new_samples'}, 'apr26_iter3_adaptive'],
            4: [{
                'eval': 'iter_4-convbert_adaptive-6057405-evaluation',
                'new_samples': 'iter_4-convbert_adaptive-6061488-new_samples'}, 'apr27_iter4_adaptive']},
        'uncertainty': {
            1: [{
                'eval': 'iter_1-convbert_uncertainty-6200469-evaluation',
                'new_samples': 'iter_1-convbert_uncertainty-6208289-new_samples'}, 'apr30_iter1_uncertainty'],
            2: [{
                'eval': 'iter_2-convbert_uncertainty-6253253-evaluation',
                'new_samples': 'iter_2-convbert_uncertainty-6293350-new_samples'}, 'may1_iter2_uncertainty'],
            3: [{
                'eval': 'iter_3-convbert_uncertainty-6318280-evaluation',
                'new_samples': 'iter_3-convbert_uncertainty-6342807-new_samples'}, 'may2_iter3_uncertainty'],
            4: [{
                'eval': 'iter_4-convbert_uncertainty-6423646-evaluation',
                'new_samples': 'iter_4-convbert_adaptive-6061488-new_samples'}, 'may3_iter4_uncertainty']
        },
        'uncertainty_uncalibrated': {
            1: [{
                'eval': 'iter_1-convbert_uncertainty_uncalibrated-6480837-evaluation',
                'new_samples': 'iter_1-convbert_uncertainty-6208289-new_samples'}, 'may6_iter1_uncertainty_uncalibrated'],
            2: [{
                'eval': 'iter_2-convbert_uncertainty_uncalibrated-6578026-evaluation',
                'new_samples': 'iter_2-convbert_uncertainty-6293350-new_samples'}, 'may7_iter2_uncertainty_uncalibrated'],
            3: [{
                'eval': 'iter_3-convbert_uncertainty_uncalibrated-6596620-evaluation',
                'new_samples': 'iter_3-convbert_uncertainty-6342807-new_samples'}, 'may8_iter3_uncertainty_uncalibrated'],
            4: [{
                'eval': 'iter_4-convbert_uncertainty_uncalibrated-6653849-evaluation',
                'new_samples': 'iter_4-convbert_adaptive-6061488-new_samples'}, 'may10_iter4_uncertainty_uncalibrated']
        }
        }
    results_dict = dict()
    labels = ['job_search', 'job_offer', 'is_hired_1mo', 'lost_job_1mo', 'is_unemployed']
    combinations_list = list(itertools.product(*[['our_method', 'adaptive', 'uncertainty', 'uncertainty_uncalibrated'], range(5), labels]))
    combinations_list = [combination for combination in combinations_list if
                         combination[:2] not in [('adaptive', 0), ('uncertainty', 0), ('uncertainty_uncalibrated', 0)]]
    selected_combinations = list(np.array_split(
        combinations_list,
        SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID])
    logger.info(f'Selected combinations: {selected_combinations}')

    for combination in selected_combinations:
        al_method, iter_nb, label = combination
        iter_nb = int(iter_nb)
        logger.info(f'Active learning method: {al_method}')
        logger.info(f'Iteration {iter_nb}')
        logger.info(f'Label: {label}')
        # load (A,B) pairs
        path_to_params = '/scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/3-model_evaluation/expansion/preliminary/calibration_dicts'
        params_dict = pickle.load(open(os.path.join(path_to_params, f'calibration_dict_{al_method}_10000_extra.pkl'), 'rb'))
        results_dict[al_method] = dict()
        results_dict[al_method][iter_nb] = dict()
        inference_folder = folder_dict[al_method][iter_nb][0]['eval']
        params = params_dict[label][inference_folder]['params']
        # load scores
        path_to_scores = os.path.join('/scratch/mt4493/twitter_labor/twitter-labor-data/data/inference',
                                      args.country_code,
                                      inference_folder, 'output', label)
        scores_df = pd.concat(
            [pd.read_parquet(path) for path in Path(path_to_scores).glob('*.parquet')]).reset_index()
        scores_df = scores_df.sort_values(by=['score'], ascending=False).reset_index(drop=True)
        # store numerators for each bootstrap model to compute standard error
        numerator_list = list()
        for count, param_pair in enumerate(params):
            if not param_pair[0] == 0:
                if count % 1000 == 0:
                    logger.info(count)
                if args.threshold == 0.5:
                    root = -param_pair[1]/param_pair[0]
                else:
                    root = optimize.brentq(func, 0, 1, args=(param_pair, args.threshold))
                numerator_list.append(scores_df.loc[scores_df['score'] > root].shape[0])
            else:
                logger.info('Discarded param pair with A=0')
        # compute and store results
        root_mean = optimize.brentq(func_mean, 0, 1, args=(params, args.threshold))
        results_dict[al_method][iter_nb][label] = dict()
        results_dict[al_method][iter_nb][label]['numerator'] = scores_df.loc[scores_df['score'] > root_mean].shape[0]
        results_dict[al_method][iter_nb][label]['sem'] = stdev(numerator_list)
    # save results dict
    output_path = f'/scratch/mt4493/twitter_labor/twitter-labor-data/data/evaluation_metrics/US/expansion/expansion_{str(SLURM_ARRAY_TASK_ID)}.json'
    with open(output_path, 'w') as f:
        json.dump(results_dict, f)
