import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from skopt import BayesSearchCV
from xgboost import XGBClassifier
from re import search
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
from config import *

warnings.filterwarnings('ignore')


def softmax(Z):
    expZ = np.exp(Z - np.max(Z))
    return expZ / expZ.sum(axis=0, keepdims=True)


df = pd.read_parquet('all_labels_with_text.parquet', engine="pyarrow")

X = np.zeros((len(regex), len(df)))
for i, row in df.iterrows():
    X[:, i] = [1 if search(r, row['text_lowercase']) is not None else 0 for r in regex]
X = pd.DataFrame(X.T)

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
)}, 'XGBClassifier': {'search': BayesSearchCV(
    estimator=XGBClassifier(
        n_jobs=-1,
        objective='binary:logistic',
        eval_metric='aucpr',
        tree_method='approx',
        use_label_encoder=False
    ),
    search_spaces={
        'learning_rate': (0.01, 1.0, 'log-uniform'),
        'max_depth': (0, 50),
        'n_estimators': (50, 100)
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
)}, 'LGBMClassifier': {'search': BayesSearchCV(
    estimator=LGBMClassifier(
        n_jobs=-1,
        objective='binary',
        silent=1
    ),
    search_spaces={
        'learning_rate': (0.01, 1.0, 'log-uniform'),
        'max_depth': (0, 50),
        'n_estimators': (50, 100)
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

for column_name in column_names:
    lbl = LabelEncoder()
    mask = df[column_name].isin(labels)
    y_train = lbl.fit_transform(df[mask][column_name])
    X_train = X[mask]
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    for i, model_name in enumerate(bayes_cv_tuners):
        bayes_cv_tuners[model_name]['search'].fit(X_train, y_train)
        model = bayes_cv_tuners[model_name]['search'].best_estimator_
        bayes_cv_tuners[model_name][f'model_on_{column_name}'] = model

        preds = model.predict_proba(X_test)[:, 1]
        report = classification_report(y_test, model.predict(X_test), output_dict=True)
        bayes_cv_tuners[model_name][f'report_on_{column_name}'] = report

        print(
            f"{model_name} on {column_name} Best ROC-AUC:"
            f"{np.round(bayes_cv_tuners[model_name]['search'].best_score_, 4)}")

        bayes_cv_tuners[model_name][f'roc_auc_score_on_{column_name}'] = roc_auc_score(y_test, preds)
        print('The baseline AUC score on the test set is {:.4f}.'.format(
            bayes_cv_tuners[model_name][f'roc_auc_score_on_{column_name}']))

        fpr, tpr, thresholds = roc_curve(y_test, preds)
        plt.plot(fpr, tpr, label="{} - {} (area = {:.4f}).".format(model_name, column_name, bayes_cv_tuners[model_name][
            f'roc_auc_score_on_{column_name}']))

    # Plot Base Rate ROC
    plt.plot([0, 1], [0, 1], label=f'Base Rate on {column_name} k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Graph on {column_name}')
    plt.legend(loc="lower right")
    plt.savefig(f'figures/auc_roc_{column_name}')
    plt.show()

    # get features
    Z = bayes_cv_tuners['LGBMClassifier'][f'model_on_{column_name}'].feature_importances_
    # normalize them and softmax them
    Z = softmax(Z / Z.max())
    features_importance = {'LGBMClassifier': Z,
                           'XGBClassifier': bayes_cv_tuners['XGBClassifier'][
                               f'model_on_{column_name}'].feature_importances_}

    fig = go.Figure(data=[go.Bar(
        name=model_name,
        x=regex,
        y=features_importance[model_name],
    ) for model_name in features_importance],
        layout={'title': f'Feature Importance for {column_name}',
                'xaxis_title': "Regex",
                'yaxis_title': "Feature Importance",
                'legend_title': "Classifier",
                })
    fig.update_layout(barmode='group')
    fig.write_html(f'figures/feature_importance_{column_name}.html')
    # fig.show()

pickle.dump(bayes_cv_tuners, open('bayes_cv_tuners.pkl', 'wb'))
