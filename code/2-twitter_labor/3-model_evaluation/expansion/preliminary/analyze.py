import pickle
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import pickle
import numpy as np

bayes_cv_tuners = pickle.load(open('bayes_cv_tuners.pkl', 'rb'))
column_names = ['is_unemployed', 'lost_job_1mo', 'job_search', 'is_hired_1mo', 'job_offer']
classifiers = ['LogisticRegression']  # list(bayes_cv_tuners.keys())

score = {}
for model_name in classifiers:
    score[model_name] = [bayes_cv_tuners[model_name][f'roc_auc_score_on_{column_name}'] for column_name in column_names]

score['Twitter labor'] = [0.965, 0.959, 0.980, 0.976, 0.985]

fig = go.Figure(data=[go.Scatter(
    name=model_name,
    x=column_names,
    y=score[model_name],
) for model_name in score],
    layout={'title': f'Classifier AUC score per label',
            'xaxis_title': "Labels",
            'yaxis_title': "AUC score",
            'legend_title': "Classifier",
            })
fig.update_layout(barmode='group')
fig.write_html(f'figures/classifier_auc_score_per_label.html')
fig.show()

score = {}
results_df = pd.DataFrame()
for model_name in classifiers:
    score[model_name] = [{
        'Accuracy': bayes_cv_tuners[model_name][f'report_on_{column_name}']['accuracy'],
        'F1-score': bayes_cv_tuners[model_name][f'report_on_{column_name}']['macro avg']['f1-score'],
        'Precision': bayes_cv_tuners[model_name][f'report_on_{column_name}']['macro avg']['precision'],
        'Recall': bayes_cv_tuners[model_name][f'report_on_{column_name}']['macro avg']['recall'],
        'ROC-AUC': bayes_cv_tuners[model_name][f'roc_auc_score_on_{column_name}'],
        'model_name': model_name,
        'label': column_name
    } for column_name in column_names]
    results_df = results_df.append(pd.DataFrame(score[model_name]))

fig = px.bar(
    data_frame=results_df.set_index('label'),
    hover_name='model_name',
    title='Metric scores per label',
    labels={
        'label': 'Classification Label',
        'value': 'Score'},
)
fig.update_layout(barmode='group')
fig.update_traces(textposition='auto')
fig.write_html(f'figures/classifier_metrics.html')

for column_name in column_names:
    pickle.dump(bayes_cv_tuners['LogisticRegression'][f'model_on_{column_name}'],
                open(f'models/lr_{column_name}.pkl', 'wb'))

x = range(10)
y = bayes_cv_tuners['LogisticRegression']['search'].cv_results_
plt.plot(x, y['mean_test_score'], 'or')
plt.fill_between(x, np.asarray(y['mean_test_score']) - np.asarray(y['std_test_score']),
                 np.asarray(y['mean_test_score']) + np.asarray(y['std_test_score']), color='gray', alpha=0.2)
plt.show()

for i in range(5):
    plt.errorbar(x, y[f'split{i}_test_score'], yerr=np.asarray(y['std_test_score']), uplims=True, lolims = True,
                 label=f'split{i}_test_score')
plt.title('Logistic Regression test score by splits')
plt.legend()
plt.savefig(f'figures/logistic_regression_test_score_by_splits')
plt.show()
