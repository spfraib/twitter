import pandas as pd
from typing import List

from config import *
import pickle
from sklearn.linear_model import LogisticRegression
import numpy as np
from re import search
from tqdm import tqdm
from glob import glob
import plotly.express as px


def run():
    iter_name = iter_names[0]
    df = pd.read_parquet('data/for_boris.parquet', engine="pyarrow")
    batch_size = 500
    models: List[LogisticRegression] = [pickle.load(open(f'models/{iter_name}/{label}.pkl', 'rb')) for label in column_names]
    preds = np.zeros((len(df), len(column_names)))

    for j in tqdm(range(0, len(df), batch_size)):
        X = np.zeros((len(new_regex), min(batch_size, len(df) - j)))
        for i, t in enumerate(df.iloc[j:j+batch_size]['text_lower']):
            X[:, i] = [1 if search(r, t) is not None else 0 for r in new_regex]
        X_train = pd.DataFrame(X.T)
        for idx, model in enumerate(models):
            preds[j:j+batch_size, idx] = model.predict_proba(X_train)[:, 1] > 0.5

    for idx, column_name in enumerate(column_names):
        df[column_name] = preds[:, idx]

    df.to_parquet('data/lr_prediction.parquet', engine="pyarrow")

    print(df[column_names].sum())

    for v in ['Train', 'Test']:
        res = {}
        for idx, label in enumerate(column_names):
            res[label] = {'Exists': 0, 'Predicted': 0, 'total': 0}
            X = pd.read_csv(glob(f'data/{iter_name}/train_test/{v.lower()}_{label}_random.csv')[0])
            X.dropna(how='any', inplace=True)
            X.reset_index(drop=True, inplace=True)
            y_test = np.asarray(X[label])

            # X['text_lowercase'] = X.text.map(str.lower)

            x = np.zeros((len(new_regex), len(X)))
            for i, row in X.iterrows():
                x[:, i] = [1 if search(r, row['text_lowercase']) is not None else 0 for r in new_regex]
            X = pd.DataFrame(x.T)

            preds = models[idx].predict(X)

            res[label]['Exists'] = y_test.sum()
            res[label]['Predicted'] = preds.sum()
            res[label]['total'] = len(y_test)

            # print(f'{v} set: {label}: {y_test.sum()} out of {len(y_test)}')
            # print(f'Predicted: {label}: {preds.sum()} out of {len(preds)}')

        res = pd.DataFrame(res).transpose()
        res.to_csv(f'{v.lower()}_predict.csv')
        print(res)

    df = pd.concat([pd.DataFrame(model.coef_[0], index=new_regex) for model in models], axis=1)
    df.columns = column_names

    fig = px.bar(df, orientation='h')
    fig.update_layout(barmode='group')
    fig.write_html('Coeff Bar Plot Logistic Regression.html')
