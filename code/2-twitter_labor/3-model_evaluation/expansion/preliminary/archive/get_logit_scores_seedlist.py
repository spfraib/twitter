import pickle
from re import search
import numpy as np
import pandas as pd
from config import *
import argparse
from glob import glob
from pathlib import Path
import os
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    return args


def predict(df, model, class_label, regex, column_name='text_lowercase'):
    """
    Given a DataFrame containing the tweets, it will check each tweet in `column_name`
    and will return a Data frame of shape (len(df), len(models)) where each column corresponds to a certain model
    :param df: dataframe to predict labels
    :param model: a loaded model to predict with
    :param class_label: name of the label to predict. Will serve as a column name in the output
    :param column_name: column name from the dataframe to predict on
    :return: a dataframe of shape (len(df), 1) where the column name correspond to the class_label
    """
    X = np.zeros((len(regex), len(df)))
    for i, row in df.iterrows():
        X[:, i] = [1 if search(r, row[column_name]) is not None else 0 for r in regex]
    X = pd.DataFrame(X.T)
    return pd.DataFrame({class_label: model.predict_proba(X)[:, 1]})


def count(preds, threshold=0.5):
    """
    given a df with column score, will calculate how many tweets got a score better than threshold
    :param preds: df returned from predict
    :param threshold: threshold to test on, default is 0.5
    :return: a count of how many tweets got a score > threshold
    """
    return (preds > threshold).sum()[0]


if __name__ == '__main__':
    args = get_args_from_command_line()
    input_path = '/scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/3-model_evaluation/expansion/preliminary'
    random_path = '/scratch/mt4493/twitter_labor/twitter-labor-data/data/random_samples/random_samples_splitted/US/evaluation'
    random_df = pd.concat([pd.read_parquet(path) for path in Path(random_path).glob('*.parquet')])
    random_df['text_lowercase'] = random_df['text'].apply(lambda x: x.lower())
    output_path = '/scratch/mt4493/twitter_labor/twitter-labor-data/data/random_samples/random_samples_splitted/US/evaluation_seedlist_logit_scores'
    results_dict = dict()
    for label in column_names:
        logger.info(f'Label: {label}')
        path_model = f'{input_path}/models/jan5_iter0/{label}.pkl'
        model = pickle.load(open(path_model, 'rb'))
        preds = predict(random_df, model, label, regex=regex).reset_index(drop=True)
        final_df = random_df.join(preds)
        final_df.to_parquet(os.path.join(output_path, f'{label}.parquet'), index=False)
        results_dict[label] = count(preds, args.threshold)
    logger.info(results_dict)