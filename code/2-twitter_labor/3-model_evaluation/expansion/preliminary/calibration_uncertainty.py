import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from config import *
import pickle
import warnings
import os
import re

warnings.filterwarnings('ignore')

plot_a_b = False
plot_score_calib = False
single = True

# number of times to sample data
num_samples = 10000
print(f'Calibrating with {num_samples}')
path_data = '/home/manuto/Documents/world_bank/bert_twitter_labor/twitter-labor-data/data/active_learning/evaluation_inference/US'
fig_path = '/home/manuto/Documents/world_bank/bert_twitter_labor/code/twitter/code/2-twitter_labor/3-model_evaluation/expansion/preliminary'
params_dict = {}

iter_names_list = ['iter_1-convbert_uncertainty-6200469-evaluation']
iter_number = int(re.findall('iter_(\d)', iter_names_list[0])[0])
for label in column_names:
    params_dict[label] = {}
    for i, iter_name in enumerate(iter_names_list):
        # load data
        df = pd.read_csv(f'{path_data}/{iter_name}/{label}.csv')
        # df['log_score'] = np.log10(df['score'])
        params = []
        params_dict[label][iter_name] = {}
        # get the positives
        positives_df = df[df['class'] == 1]
        # negatives
        negatives_df = df[df['class'] == 0]
        # sample min(len(positives_df), len(negatives_df)) rows from the data without replacement
        for negative_df in [negatives_df.sample(n=min(len(positives_df), len(negatives_df)), replace=False) for _ in
                            range(num_samples)]:
            dp = positives_df.sample(n=len(positives_df), replace=True)
            dn = negative_df.sample(n=len(negative_df), replace=True)
            d = pd.concat([dp, dn])
            # build logistic regression model to fit data
            # get the scores and labels
            X = np.asarray(d['score']).reshape((-1, 1))
            y = np.asarray(d['class'])
            # perform calibration using sigmoid function with 5 cv
            try:
                model = LogisticRegression(penalty='none').fit(X, y)
            except:
                print(f'failed with {iter_name} on label {label}')
                continue
            # get all A, B for each of the model
            params.append([model.coef_[0][0], model.intercept_[0]])

        print(f'Sampled {len(positives_df)} positives for {label}, {iter_name}')
        # calculate the calibrated score:  avg(logit(ax+b))
        all_calibrated_scores = [1 / (1 + np.exp(-(param[0] * df['score'] + param[1]))) for param in params]
        df['Calibrated score'] = np.mean(all_calibrated_scores, axis=0)
        params_dict[label][iter_name]['params'] = params
        if single:
            # plot A, B obtained from logistic regression
            if plot_a_b and len(params) >= 10:
                plt.plot(range(len(params)), [param[0] for param in params], label='A')
                plt.plot(range(len(params)), [param[1] for param in params], label='B')
                plt.title(f'A and B on iter_{i} - {label} with {num_samples} samples')
                plt.xlabel("iteration")
                plt.ylabel("value")
                plt.legend(loc='best')
                plt.savefig(f'figures/{iter_name}/{label}_a_b_dist_{num_samples}')
                plt.show()
            # plot score / calibrated score ration
            if plot_score_calib:
                plt.plot(df['score'], df['Calibrated score'])
                plt.title(f'Platt scaling on iter_{i} - {label}')
                plt.xlabel("Score of BERT")
                plt.ylabel("Calibrated probability")
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                plt.savefig(f'figures/{iter_name}/{label}_calibration')
                plt.show()
            # get the rank
            rank = [df.iloc[i]['rank'] for i in range(0, len(df), 10)]
            # get the mean score of BERT
            values = [np.mean(df[i:i + 10]['score']) for i in range(0, len(df), 10)]
            # get the share of positives
            positives = [np.mean(df[i:i + 10]['class']) for i in range(0, len(df), 10)]
            # get the Platt calibrated score mean
            calibrated = [np.mean(df[i:i + 10]['Calibrated score']) for i in range(0, len(df), 10)]
            # plot
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(rank, values, label='Bert score', marker='x')
            ax.plot(rank, values)
            ax.scatter(rank, positives, label='share of positive labels', marker='x')
            ax.plot(rank, positives)
            ax.scatter(rank, calibrated, marker='x', label=f'Platt calibrated score iter_{i} (n={len(positives_df)})')
            ax.plot(rank, calibrated)
            ax.set_xscale('log')
            ax.axvspan(0, 1e4, alpha=0.05, color='gray')
            ax.axvline(10000,linewidth=.75,color='k',linestyle='--')
            ax.set_title(f'Calibrated Score to rank of {label} with {num_samples} samples')
            ax.set_xlabel('Rank of predicted score')
            ax.set_ylabel("Calibrated score")
            ax.legend(loc='best')
            if not os.path.exists(f'{fig_path}/figures/{iter_name}'):
                os.makedirs(f'{fig_path}/figures/{iter_name}')
            plt.savefig(f'{fig_path}/figures/{iter_name}/{label}_mean_score_single_{num_samples}_our_method')
            # plt.show()
        #
        # else:
        #     rank = [df.iloc[i]['rank'] for i in range(0, len(df), 10)]
        #     calibrated = [np.mean(df[i:i + 10]['Calibrated score']) for i in range(0, len(df), 10)]
        #     plt.scatter(rank, calibrated, marker='x', label=f'Platt score iter_{i} (n={len(positives_df)})')
        #     line = plt.plot(rank, calibrated)
        #     # for scores in all_calibrated_scores:
        #     # std = np.std(all_calibrated_scores, axis=0)
        #     # plt.fill_between(df['rank'], df['score'] - std, df['score'] + std, alpha=0.1, color=line[0]._color)
        #     plt.axvspan(0, 1e4, alpha=0.05, color='gray')
        #     plt.axvline(1e4, ls='--', color='black')
        #     plt.title(f'Calibrated Score to rank of {label} with {num_samples} samples')
        #     plt.xlabel('Rank of predicted score')
        #     plt.ylabel("Calibrated score")
        #     plt.legend(loc='best')
        #     plt.xscale('log')
        #     plt.savefig(f'figures/{iter_name}/{label}_mean_score_combined_{num_samples}')
        # # save the variables to a dict to save for later
        # params_dict[label][iter_name]['params'] = params
        # save the results
        # df.to_csv(f'data/{iter_name}/calibrate/{label}.csv', index=False)

    if not single:
        plt.savefig(f'figures/{label}/iter_predicted_score_{num_samples}')
        plt.show()
# params_dict =
# {'is_hired_1mo': {'jan5_iter0': {'params': [[1568.9405631426648, 20.02107572603093],
#                                             [2198.0807588968146, 28.531457694715392],
#                                             [2495.7452000504454, 32.84797611983793],
#                                             [1573.6622693808868, 20.875867126052224]]},
#                   'feb22_iter1': {'params': [[31.39730728272755, 2.053297533949996],
#     [31.251631996069698, 1.4747238563491978],
#     [74.02178341769871, 1.5411200969559047],
#     [39.02320593796812, 1.7064392251901723]]}},
# 'is_unemployed': {'jan5_iter0': {'params': [[4608.914326196402, 23.71499747450902],...]},
#                   ...},
#  ...}

pickle.dump(params_dict, open(f'{fig_path}/calibration_dicts/calibration_dict_uncertainty_{num_samples}_iter{iter_number}.pkl', 'wb'))
