import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from config import *
import pickle
import warnings
import os

warnings.filterwarnings('ignore')

plot_a_b = False
plot_score_calib = False
single = False


path_data = '/home/manuto/Documents/world_bank/bert_twitter_labor/twitter-labor-data/data/active_learning/evaluation_inference/US'
fig_path = '/home/manuto/Documents/world_bank/bert_twitter_labor/code/twitter/code/2-twitter_labor/3-model_evaluation/expansion/preliminary'
params_dict = {}
for label in column_names:
    print(label)
    for i, iter_name in enumerate(['iter_0-convbert-969622-evaluation']):
        # load data
        df = pd.read_csv(f'{path_data}/{iter_name}/{label}.csv')
        # df['log_score'] = np.log10(df['score'])
        # build logistic regression model to fit data
        # get the scores and labels
        X = np.asarray(df['score']).reshape((-1, 1))
        y = np.asarray(df['class'])
        # perform calibration using sigmoid function with 5 cv
        try:
            model = LogisticRegression(penalty='none').fit(X, y)
        except:
            print(f'failed with {iter_name} on label {label}')
            continue
        # get all A, B for each of the model
        params_dict[label] = [model.coef_[0][0], model.intercept_[0]]

print(params_dict)

