import numpy as np
from scipy import optimize
import pickle
import os

def func(x, params):
    all_calibrated_scores = [1 / (1 + np.exp(-(param[0] * x + param[1]))) for param in params]
    return np.mean(all_calibrated_scores, axis=0) - 0.5

def calibrate(x, params):
    all_calibrated_scores = [1 / (1 + np.exp(-(param[0] * x + param[1]))) for param in params]
    return np.mean(all_calibrated_scores, axis=0)

folder_dict = {
    0: [{
            'eval': 'iter_0-convbert-969622-evaluation',
            'new_samples': 'iter_0-convbert-1122153-new_samples'}, 'jan5_iter0'],
    1: [{
            'eval': 'iter_1-convbxert-3050798-evaluation',
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

path_to_params = '/home/manuto/Documents/world_bank/bert_twitter_labor/code/twitter/code/2-twitter_labor/3-model_evaluation/expansion/preliminary'
params_dict = pickle.load(open(os.path.join(path_to_params, 'calibration_dict_our_method_10000.pkl'), 'rb'))
inference_folder = folder_dict[0][0]['eval']

for label in ['is_hired_1mo', 'is_unemployed', 'lost_job_1mo', 'job_search', 'job_offer']:
    print(label)
    params = params_dict[label][inference_folder]['params']
    root = optimize.brentq(func, 0, 1, args=(params))
    print(f'Root: {root}')
    print(f'Proba with score=root: {calibrate(root, params)}')




