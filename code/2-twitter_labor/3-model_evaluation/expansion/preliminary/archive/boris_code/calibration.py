import pandas as pd
import numpy as np
from netcal.scaling import LogisticCalibration
import matplotlib.pyplot as plt
from re import search
from config import *
import pickle


def normalize(df):
    min_x = 1
    max_x = 1e8
    return df['rank'].apply(lambda x: (x - min_x) / (max_x - min_x))


df = pd.read_csv('data/jan5_iter0/calibrate/is_unemployed.csv')

matched = np.asarray(df['class'])
confidences = df['score']
relative_x_position = df.reset_index()['index'].rank(pct=True)
relative_x_position_normalized = normalize(df)

input = np.stack((confidences, relative_x_position), axis=1)

lr = LogisticCalibration(detection=True)        # flag 'detection=True' is mandatory for this method
lr.fit(input, matched)
df['Calibrated score'] = lr.transform(input)

input = np.stack((confidences, relative_x_position_normalized), axis=1)

lr = LogisticCalibration(detection=True)        # flag 'detection=True' is mandatory for this method
lr.fit(input, matched)
df['Calibrated normalized score'] = lr.transform(input)

df[['score', 'Calibrated score', 'Calibrated normalized score']].plot()
plt.title('Plat scaling on jan5_iter0 - is_unemployed')
plt.xlabel("Relative rank")
plt.ylabel("Score")
plt.show()

X_train = pd.read_csv(f'data/jan5_iter0/train_test/val_is_unemployed.csv')
X_train.dropna(how='any', inplace=True)
X_train.reset_index(drop=True, inplace=True)
y_train = np.asarray(X_train['class'])

X_train['text_lowercase'] = X_train.text.map(str.lower)

X = np.zeros((len(regex), len(X_train)))
for i, row in X_train.iterrows():
    X[:, i] = [1 if search(r, row['text_lowercase']) is not None else 0 for r in regex]
X_train = pd.DataFrame(X.T)

model = pickle.load(open('results/jan5_iter0/bayes_cv_tuners_is_unemployed.pkl', 'rb'))['LogisticRegression']['model_is_unemployed']
matched = np.asarray(y_train)

X_train['Score'] = model.predict_proba(X_train)[:, 1]
confidences = X_train['Score']

X_train['rank'] = X_train['Score'].rank(pct=True)
relative_x_position = X_train['rank']

input = np.stack((confidences, relative_x_position), axis=1)

lr = LogisticCalibration(detection=True)        # flag 'detection=True' is mandatory for this method
lr.fit(input, matched)
X_train['Calibrated Score'] = lr.transform(input)

X_train.sort_values(by=['rank'], ascending=False).set_index(keys='rank', drop=True)[['Score', 'Calibrated Score']].plot()
plt.title('Plat scaling on jan5_iter0 - is_unemployed Logistic regression')
plt.xlabel("Relative rank")
plt.ylabel("Score")
plt.show()
