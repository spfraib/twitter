import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from config import *
import pickle

# number of times to sample data
num_samples = 1000
params_dict = {}
rus = RandomUnderSampler(random_state=42, replacement=True)

for iter_name in iter_names:
    params_dict[iter_name] = {}
    print(f'Calibrating {iter_name}')
    for label in column_names:
        #load data
        df = pd.read_csv(f'data/{iter_name}/calibrate/{label}.csv')

        params = []
        params_dict[iter_name][label] = {}
        # sample 300 rows from the data with replacement
        for d in [df.sample(n=300, replace=True, random_state=42) for _ in range(num_samples)]:
            # build logistic regression model to fit data
            model = LogisticRegression(random_state=42)
            # get the scores and labels
            X = np.asarray(d['score']).reshape((-1, 1))
            y = np.asarray(d['class'])
            # under sample
            X, y = rus.fit_resample(X, y)
            # perform calibration using sigmoid function with 5 cv
            clf = CalibratedClassifierCV(model)
            clf.fit(X, y)
            # get all A, B for each of the model and average them
            params.append(
                np.mean(
                    [[cc.calibrators_[0].a_, cc.calibrators_[0].b_] for cc in clf.calibrated_classifiers_],
                    axis=0)
            )

        print(f'Sampled {np.sum(y)} of each label for {label}')
        # average all the variables
        final_params = np.mean(params, axis=0)
        # calculate the calibrated score
        df['Calibrated score'] = 1 / (1 + np.exp(final_params[0] * df['score'] + final_params[1]))
        # plot the model score in comparison to the model score
        df[['score', 'Calibrated score']].plot()
        plt.title(f'Plat scaling on {iter_name} - {label}')
        plt.xlabel("Relative rank")
        plt.ylabel("Score")
        plt.savefig(f'figures/{iter_name}/{label}_calibration')
        plt.show()
        # save the variables to a dict to save for later
        params_dict[iter_name][label]['A'] = final_params[0]
        params_dict[iter_name][label]['B'] = final_params[1]
# pickle the dict to use later
# params_dict =
# {
# 'jan5_iter0': {
#   'is_unemployed': {'A': -23594.40989080293,'B': -2486.059432789291},
#   'lost_job_1mo':  {'A': -12.60080639489828, 'B': 2.2142054175197066},
#   'job_search':    {'A': -12051.593057311928, 'B': 10.695008923562453},
#   'is_hired_1mo':  {'A': -2.282760559888111, 'B': 0.29217173198546853},
#   'job_offer':     {'A': -15235.177183379248, 'B': 2.2260417469324585}
# },...
pickle.dump(params_dict, open('calibration_dict.pkl', 'wb'))
