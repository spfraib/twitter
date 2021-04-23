import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from config import *
import pickle

num_samples = 1000
params_dict = {}
rus = RandomUnderSampler(random_state=42, replacement=True)

for iter_name in iter_names:
    params_dict[iter_name] = {}
    print(f'Calibrating {iter_name}')
    for label in column_names:
        df = pd.read_csv(f'data/{iter_name}/calibrate/{label}.csv')

        params = []
        params_dict[iter_name][label] = {}

        for d in [df.sample(n=300, replace=True, random_state=42) for _ in range(num_samples)]:
            model = LogisticRegression(random_state=42)
            X = np.asarray(d['score']).reshape((-1, 1))
            y = np.asarray(d['class'])
            X, y = rus.fit_resample(X, y)
            clf = CalibratedClassifierCV(model)
            clf.fit(X, y)

            params.append(
                np.mean(
                    [[cc.calibrators_[0].a_, cc.calibrators_[0].b_] for cc in clf.calibrated_classifiers_],
                    axis=0)
            )

        print(f'Sampled {np.sum(y)} of each label for {label}')

        final_params = np.mean(params, axis=0)
        df['Calibrated score'] = 1 / (1 + np.exp(final_params[0] * df['score'] + final_params[1]))
        df[['score', 'Calibrated score']].plot()
        plt.title(f'Plat scaling on {iter_name} - {label}')
        plt.xlabel("Relative rank")
        plt.ylabel("Score")
        plt.savefig(f'figures/{iter_name}/{label}_calibration')
        #plt.show()

        params_dict[iter_name][label]['A'] = final_params[0]
        params_dict[iter_name][label]['B'] = final_params[1]

pickle.dump(params_dict, open('calibration_dict.pkl', 'wb'))
