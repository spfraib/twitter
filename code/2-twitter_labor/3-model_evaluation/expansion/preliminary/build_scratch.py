from typing import List

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

from config import *
from re import search
import numpy as np
from glob import glob
import pickle
import matplotlib.pyplot as plt
from lr_prediction import run

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_parquet('data/for_boris_2.parquet')
label_df = pd.read_parquet('data/all_labels_with_text.parquet')
df = df[~df['ngram'].isin(['i_fired', 'firedme', 'i_unemployed', 'i_jobless', 'i_not_working'])]

for r in regex:
    df[r] = df['ngram'].apply(lambda ngram: 1 if search(r, ngram) is not None else 0)

new_regex = list(df['ngram'].unique())

df['regex_num'] = df['ngram'].map(new_regex.index)
res = {}
reg_train = {}
reg_test = {}
for column_name in column_names:
    reg_train[column_name] = {}
    reg_test[column_name] = {}

    res[column_name] = {'train_positives': 0, 'test_positives': 0, 'total_train': 0, 'total_test': 0}
    mask = label_df[column_name].isin(labels)
    X_train = label_df[mask][['tweet_id', 'text_lowercase', column_name]]
    X_train = pd.merge(X_train, df[['tweet_id', 'regex_num']], on='tweet_id').drop_duplicates(subset=['tweet_id'])
    X_train[column_name] = X_train[column_name].apply(lambda label: 1 if label == 'yes' else 0)
    g = X_train.groupby('regex_num')

    train = []
    test = []
    for reg_num, sdf in g:
        strain, stest = train_test_split(sdf, test_size=0.33)

        reg_train[column_name][new_regex[reg_num]] = len(strain)
        reg_test[column_name][new_regex[reg_num]] = len(stest)

        if sdf[column_name].sum() >= 3:
            train.append(strain)
            test.append(stest)

    train = pd.concat(train)
    train.to_csv(f'data/jan5_iter0/train_test/train_{column_name}.csv', index=False)

    test = pd.concat(test)
    test.to_csv(f'data/jan5_iter0/train_test/test_{column_name}.csv', index=False)

    res[column_name]['train_positives'] = train[column_name].sum()
    res[column_name]['test_positives'] = test[column_name].sum()
    res[column_name]['total_train'] = len(train)
    res[column_name]['total_test'] = len(test)

res = pd.DataFrame(res).transpose()
res.to_csv('train_test_ratio.csv')

reg_train = pd.DataFrame(reg_train)
reg_train.columns = reg_train.columns + '_train'

reg_test = pd.DataFrame(reg_test)
reg_test.columns = reg_test.columns + '_test'

reg = pd.concat([reg_train, reg_test], axis=1)
reg = reg.reindex(sorted(reg.columns), axis=1)

reg.to_csv('regex_count per set.csv')
print(res)
print(reg)

reports = []
iter_name = iter_names[0]
for label in column_names:
    train_df = pd.read_csv(glob(f'data/{iter_name}/train_test/train_{label}.csv')[0])
    train_df.dropna(how='any', inplace=True)
    test_df = pd.read_csv(glob(f'data/{iter_name}/train_test/test_{label}.csv')[0])
    test_df.dropna(how='any', inplace=True)

    g_train = train_df.groupby('regex_num')
    g_test = test_df.groupby('regex_num')

    train = []
    test = []
    for regex_num, sdf in g_train:
        positives_df = sdf[sdf[label] == 1]
        negatives = sdf[sdf[label] == 0]

        negatives_df = negatives.sample(n=min(len(positives_df), len(negatives)), replace=False, random_state=42)
        train.append(pd.concat([positives_df, negatives_df]))

    for regex_num, sdf in g_test:
        positives_df = sdf[sdf[label] == 1]
        negatives = sdf[sdf[label] == 0]

        negatives_df = negatives.sample(n=min(len(positives_df), len(negatives)), replace=False, random_state=42)
        test.append(pd.concat([positives_df, negatives_df]))

    train_df = pd.concat(train)
    test_df = pd.concat(test)
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    y_train = np.asarray(train_df[label])
    y_test = np.asarray(test_df[label])

    train_df.to_csv(f'data/jan5_iter0/train_test/train_{label}_random.csv', index=False)
    test_df.to_csv(f'data/jan5_iter0/train_test/test_{label}_random.csv', index=False)

    # train_df = pd.read_csv(f'data/jan5_iter0/train_test/train_{label}_random.csv')
    # test_df = pd.read_csv(f'data/jan5_iter0/train_test/test_{label}_random.csv')

    y_train = np.asarray(train_df[label])
    y_test = np.asarray(test_df[label])

    X = np.zeros((len(regex), len(train_df)))
    for i, row in train_df.iterrows():
        X[:, i] = [1 if search(r, row['text_lowercase']) is not None else 0 for r in regex]
    X_train = X.T  # pd.DataFrame(X.T)

    X = np.zeros((len(regex), len(test_df)))
    for i, row in test_df.iterrows():
        X[:, i] = [1 if search(r, row['text_lowercase']) is not None else 0 for r in regex]
    X_test = X.T  # pd.DataFrame(X.T)

    model = LogisticRegression(random_state=42, max_iter=10000)
    model.fit(X_train, y_train)
    pickle.dump(model, open(f'models/{iter_name}/{label}.pkl', 'wb'))

    preds = model.predict_proba(X_test)[:, 1]
    report = pd.DataFrame(classification_report(y_test, model.predict(X_test), output_dict=True))
    report['class'] = label
    reports.append(report)
    auc_score = roc_auc_score(y_test, preds)
    fpr, tpr, thresholds = roc_curve(y_test, preds)

    plt.plot(fpr, tpr, label="{} - {} (area = {:.4f}).".format('LogisticRegression', label, auc_score))

plt.plot([0, 1], [0, 1], label=f'Base Rate on {iter_name} k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Graph on {iter_name}')
plt.legend(loc="lower right")
plt.savefig(f'figures/{iter_name}/auc_roc_plot')
plt.show()

reports = pd.concat(reports)
reports.to_csv('reports for lr.csv')

run()