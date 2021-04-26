import numpy as np
import pandas as pd
import os
import argparse
import pickle
from pathlib import Path

def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str, default='US')
    parser.add_argument("--set", type=str)

    args = parser.parse_args()
    return args

def calibrate(score, params):
    return np.mean([1 / (1 + np.exp(-(param[0] * score + param[1]))) for param in params], axis=0)

if __name__ == '__main__':
    args = get_args_from_command_line()
    path_to_params = '/scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/3-model_evaluation/expansion/preliminary'
    params_dict = pickle.load(open(os.path.join(path_to_params, 'calibration_dict.pkl'), 'rb'))
    folder_dict = {
        0: [{'eval': 'iter_0-convbert-969622-evaluation', 'new_samples': 'iter_0-convbert-1122153-new_samples'}, 'jan5_iter0'],
        1: [{'eval': 'iter_1-convbert-3050798-evaluation', 'new_samples': 'iter_1-convbert-3062566-new_samples'}, 'feb22_iter1'],
        2: [{'eval': 'iter_2-convbert-3134867-evaluation', 'new_samples': 'iter_2-convbert-3139138-new_samples'},  'feb23_iter2'],
        3: [{'eval': 'iter_3-convbert-3174249-evaluation', 'new_samples': 'iter_3-convbert-3178321-new_samples'}, 'feb25_iter3'],
        4: [{'eval': 'iter_4-convbert-3297962-evaluation', 'new_samples': 'iter_4-convbert-3308838-new_samples'}, 'mar1_iter4']}


    for label in ['is_hired_1mo', 'lost_job_1mo', 'is_unemployed', 'job_search', 'job_offer']:
        for iter in range(5):
            inference_folder = folder_dict[iter][0][args.set]
            data_folder = folder_dict[iter][1]
            params = params_dict[label][data_folder]['params']

            path_to_scores = os.path.join('/scratch/mt4493/twitter_labor/twitter-labor-data/data/inference', args.country_code,
                                          inference_folder, 'output', label)
            scores_df = pd.concat([pd.read_parquet(path) for path in Path(path_to_scores).glob('*.parquet')])
            scores_df['calibrated_score'] = scores_df['score'].apply(lambda x: calibrate(x, params=params))
            output_path = os.path.join(Path(path_to_scores).parents[1], 'calibrated_output')
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            scores_df.to_parquet(os.path.join(output_path, 'calibrated_scores.parquet'), index=False)
