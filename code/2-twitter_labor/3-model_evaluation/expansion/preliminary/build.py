import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from skopt import BayesSearchCV
from re import search
import matplotlib.pyplot as plt
import pickle
import warnings
from config import *
from glob import glob

warnings.filterwarnings('ignore')


def softmax(Z):
    expZ = np.exp(Z - np.max(Z))
    return expZ / expZ.sum(axis=0, keepdims=True)


for label in column_names:
    for iter_name in iter_names:
        X_train = pd.read_csv(glob(f'data/{iter_name}/train_test/train_{label}.csv')[0])
        X_train.dropna(how='any', inplace=True)
        X_train.reset_index(drop=True, inplace=True)
        y_train = np.asarray(X_train['class'])
        X_test = pd.read_csv(glob(f'data/{iter_name}/train_test/val_{label}.csv')[0])
        X_test.dropna(how='any', inplace=True)
        X_test.reset_index(drop=True, inplace=True)
        y_test = np.asarray(X_test['class'])

        X_train['text_lowercase'] = X_train.text.map(str.lower)
        X_test['text_lowercase'] = X_test.text.map(str.lower)

        X = np.zeros((len(regex), len(X_train)))
        for i, row in X_train.iterrows():
            X[:, i] = [1 if search(r, row['text_lowercase']) is not None else 0 for r in regex]
        X_train = pd.DataFrame(X.T)

        X = np.zeros((len(regex), len(X_test)))
        for i, row in X_test.iterrows():
            X[:, i] = [1 if search(r, row['text_lowercase']) is not None else 0 for r in regex]
        X_test = pd.DataFrame(X.T)

        bayes_cv_tuners = {'LogisticRegression': {'search': BayesSearchCV(
            estimator=LogisticRegression(
                random_state=42
            ),
            search_spaces={
                'solver': ['liblinear', 'saga'],
                'penalty': ['l1', 'l2'],
                'fit_intercept': [True, False]
            },
            scoring='roc_auc',
            cv=StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=42
            ),
            n_jobs=-1,
            n_iter=ITERATIONS,
            verbose=0,
            refit=True,
            random_state=42
        )}}

        bayes_cv_tuners['LogisticRegression']['search'].fit(X_train, y_train)
        model = bayes_cv_tuners['LogisticRegression']['search'].best_estimator_
        bayes_cv_tuners['LogisticRegression'][f'model_{label}'] = model

        pickle.dump(model, open(f'models/{iter_name}/{label}.pkl', 'wb'))

        preds = model.predict_proba(X_test)[:, 1]
        report = classification_report(y_test, model.predict(X_test), output_dict=True)
        bayes_cv_tuners['LogisticRegression'][f'report_{label}'] = report

        print(f"{iter_name}:\n"
              f"LogisticRegression on {label} Best ROC-AUC:"
              f"{np.round(bayes_cv_tuners['LogisticRegression']['search'].best_score_, 4)}")

        bayes_cv_tuners['LogisticRegression'][f'roc_auc_score_{label}'] = roc_auc_score(y_test, preds)
        print('The baseline AUC score on the test set is {:.4f}.'.format(
            bayes_cv_tuners['LogisticRegression'][f'roc_auc_score_{label}']))

        fpr, tpr, thresholds = roc_curve(y_test, preds)
        plt.plot(fpr, tpr, label="{} - {} (area = {:.4f}).".format('LogisticRegression', iter_name,
                                                                   bayes_cv_tuners['LogisticRegression'][
                                                                       f'roc_auc_score_{label}']))

        pickle.dump(bayes_cv_tuners, open(f'results/{iter_name}/bayes_cv_tuners_{label}.pkl', 'wb'))

    # Plot Base Rate ROC
    plt.plot([0, 1], [0, 1], label=f'Base Rate on {label} k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Graph on {label}')
    plt.legend(loc="lower right")
    plt.savefig(f'figures/{label}/auc_roc_plot')
    plt.show()
