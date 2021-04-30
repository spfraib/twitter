import numpy as np
import pandas as pd
import os
import argparse
import pickle
from pathlib import Path
import logging
import socket
from numba import jit


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str, default='US')
    parser.add_argument("--set", type=str, default='new_samples')

    args = parser.parse_args()
    return args

@jit(nopython=True)
def calibrate(score_array, params_array, output_array):
    for count, score in enumerate(score_array):
        calibrated_score = 0
        for param in params_array:
            calibrated_score += 1 / (1 + np.exp(-(param[0] * score + param[1])))
        output_array[count] = calibrated_score/len(params_array)
    return output_array


if __name__ == '__main__':
    # Get args
    args = get_args_from_command_line()
    # Load params
    path_to_params = '/scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/3-model_evaluation/expansion/preliminary'
    params_dict = pickle.load(open(os.path.join(path_to_params, 'calibration_dict_our_method_10000.pkl'), 'rb'))
    folder_dict = {
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
                'new_samples': 'iter_4-convbert-3308838-new_samples'}, 'mar1_iter4']}
    for iter in [0]:
        logger.info(f'Iteration {iter}')
        for label in ['job_search', 'job_offer','is_hired_1mo', 'lost_job_1mo', 'is_unemployed']:
            logger.info(f'Label: {label}')
            inference_folder = folder_dict[iter][0][args.set]
            data_folder = folder_dict[iter][0]['eval']
            params_array = np.asarray(params_dict[label][data_folder]['params'])
            path_to_scores = os.path.join('/scratch/mt4493/twitter_labor/twitter-labor-data/data/inference',
                                          args.country_code,
                                          inference_folder, 'output', label)
            scores_df = pd.concat([pd.read_parquet(path) for path in Path(path_to_scores).glob('*.parquet')]).reset_index()
            logger.info('Loaded scores')
            score_array = scores_df['score'].to_numpy()
            output_array = np.zeros(shape=(len(score_array),))
            scores_df['calibrated_score'] = calibrate(score_array=score_array, params_array=params_array, output_array=output_array)
            output_path = os.path.join(Path(path_to_scores).parents[1], 'calibrated_output', label)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            scores_df.to_parquet(os.path.join(output_path, 'calibrated_scores.parquet'), index=False)
