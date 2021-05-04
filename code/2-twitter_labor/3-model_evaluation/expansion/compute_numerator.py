import pickle
import os
import numpy as np
import logging
from pathlib import Path
import pandas as pd
import argparse
from scipy import stats
from scipy import optimize

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str, default='US')
    parser.add_argument("--set", type=str, default='eval')
    parser.add_argument("--active_learning", type=str)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    return args

def func(x, param_pair, threshold):
    return 1 / (1 + np.exp(-(param_pair[0] * x + param_pair[1]))) - threshold


def calibrate(score, param):
    return 1 / (1 + np.exp(-(param[0] * score + param[1])))


if __name__ == '__main__':
    args = get_args_from_command_line()

    path_to_params = '/scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/3-model_evaluation/expansion/preliminary/calibration_dicts'
    if args.active_learning == 'our_method':
        filename = 'calibration_dict_our_method_10000.pkl'
    elif args.active_learning == 'adaptive':
        filename = 'calibration_dict_adaptive_10000.pkl'
    params_dict = pickle.load(open(os.path.join(path_to_params, filename), 'rb'))

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
            0: [{
                'eval': 'iter_0-convbert-969622-evaluation',
                'new_samples': 'iter_0-convbert-1122153-new_samples'}, 'jan5_iter0'],
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
    }
    results_dict = dict()
    logger.info(f'Active learning method: {args.active_learning}')
    logger.info(f'Threshold: {args.threshold}')

    for iter in range(5):
        logger.info(f'Iteration {iter}')
        results_dict[iter] = dict()
        for label in ['job_search', 'job_offer', 'is_hired_1mo', 'lost_job_1mo', 'is_unemployed']:
            logger.info(f'Label: {label}')
            inference_folder = folder_dict[args.active_learning][iter][0][args.set]
            data_folder = folder_dict[args.active_learning][iter][0]['eval']
            params = params_dict[label][data_folder]['params']
            path_to_scores = os.path.join('/scratch/mt4493/twitter_labor/twitter-labor-data/data/inference',
                                          args.country_code,
                                          inference_folder, 'output', label)
            scores_df = pd.concat(
                [pd.read_parquet(path) for path in Path(path_to_scores).glob('*.parquet')]).reset_index()
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
            results_dict[iter][label] = dict()
            results_dict[iter][label]['mean'] = np.mean(numerator_list, axis=0)
            results_dict[iter][label]['sem'] = stats.sem(numerator_list)
    print(results_dict)
