import pandas as pd
from typing import List

from config import *
import pickle
from sklearn.linear_model import LogisticRegression
import numpy as np
from re import search
from tqdm import tqdm

iter_name = iter_names[0]
df = pd.read_parquet('data/for_boris.parquet', engine="pyarrow")
batch_size = 64
models: List[LogisticRegression] = [pickle.load(open(f'models/{iter_name}/{label}.pkl', 'rb')) for label in column_names]
preds = np.zeros((len(df), len(column_names)))

for j in tqdm(range(0, len(df), batch_size)):
    X = np.zeros((len(regex), min(batch_size, len(df) - j)))
    for i, t in enumerate(df.iloc[j:j+batch_size]['text_lower']):
        X[:, i] = [1 if search(r, t) is not None else 0 for r in regex]
    X_train = pd.DataFrame(X.T)
    for idx, model in enumerate(models):
        preds[j:j+batch_size, idx] = model.predict_proba(X_train)[:, 1] > 0.5

for idx, column_name in enumerate(column_names):
    df[column_name] = preds[:, idx]

df.to_parquet('data/lr_prediction.parquet', engine="pyarrow")