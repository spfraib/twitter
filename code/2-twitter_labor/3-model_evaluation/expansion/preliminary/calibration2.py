import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from config import *
import pickle
import warnings

warnings.filterwarnings('ignore')

single = True
plot_a_b = False
plot_score_calib = False

# number of times to sample data
num_samples = 10000
params_dict = {}
for label in column_names:
    params_dict[label] = {}
    print(f'Calibrating {label}')
    for iter_name in iter_names:
        # load data
        df = pd.read_csv(f'data/{iter_name}/calibrate/{label}.csv')
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
            balanced_df = pd.concat([positives_df, negative_df])
            d = balanced_df.sample(n=len(balanced_df), replace=True)
            # build logistic regression model to fit data
            # get the scores and labels
            X = np.asarray(d['score']).reshape((-1, 1))
            y = np.asarray(d['class'])
            # perform calibration using sigmoid function with 5 cv
            model = LogisticRegression(penalty='none').fit(X, y)
            # get all A, B for each of the model
            params.append([model.coef_[0][0], model.intercept_[0]])

        print(f'Sampled {len(positives_df)} positives for {label}')
        # calculate the calibrated score:  avg(logit(ax+b))
        df['Calibrated score'] = np.mean([1 / (1 + np.exp(-(param[0] * df['score'] + param[1]))) for param in params], axis=0)

        if single:
            # plot A, B obtained from logistic regression
            if plot_a_b:
                plt.plot(range(len(params)), [param[0] for param in params], label='A')
                plt.plot(range(len(params)), [param[1] for param in params], label='B')
                plt.title(f'A and B on {iter_name} - {label}')
                plt.xlabel("iteration")
                plt.ylabel("value")
                plt.legend(loc='best')
                plt.savefig(f'figures/{iter_name}/{label}_a_b_dist')
                plt.show()
            # plot score / calibrated score ration
            if plot_score_calib:
                plt.plot(df['score'], df['Calibrated score'])
                plt.title(f'Platt scaling on {iter_name} - {label}')
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
            plt.scatter(rank, values, label='Bert score', marker='x')
            plt.plot(rank, values)
            plt.scatter(rank, positives, label='share of positive labels', marker='x')
            plt.plot(rank, positives)
            plt.scatter(rank, calibrated, marker='x', label=f'Platt calibrated score {iter_name} (n={len(positives_df)})')
            plt.plot(rank, calibrated)
            plt.axvspan(0, 1e4, alpha=0.05, color='gray')
            plt.axvline(1e4, ls='--', color='black')
            plt.title(f'Calibrated Score to rank of {label}')
            plt.xlabel('Rank of predicted score')
            plt.ylabel("Calibrated score")
            plt.legend(loc='best')
            plt.xscale('log')
            plt.savefig(f'figures/{iter_name}/{label}_mean_score_single')
            plt.show()

        else:
            rank = [df.iloc[i]['rank'] for i in range(0, len(df), 10)]
            calibrated = [np.mean(df[i:i + 10]['Calibrated score']) for i in range(0, len(df), 10)]
            plt.scatter(rank, calibrated, marker='x', label=f'Platt calibrated score {iter_name} (n={len(positives_df)})')
            plt.plot(rank, calibrated)
            plt.axvspan(0, 1e4, alpha=0.05, color='gray')
            plt.axvline(1e4, ls='--', color='black')
            plt.title(f'Calibrated Score to rank of {label}')
            plt.xlabel('Rank of predicted score')
            plt.ylabel("Calibrated score")
            plt.legend(loc='best')
            plt.xscale('log')
            plt.savefig(f'figures/{iter_name}/{label}_mean_score_combined')
        # save the variables to a dict to save for later
        params_dict[label][iter_name]['params'] = params
        # save the results
        df.to_csv(f'data/{iter_name}/calibrate/{label}.csv', index=False)

    if not single:
        plt.savefig(f'figures/{label}/iter_predicted_score')
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

pickle.dump(params_dict, open('calibration_dict.pkl', 'wb'))
