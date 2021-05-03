import pickle
from re import search
import numpy as np
import pandas as pd
from config import *
import argparse
from glob import glob


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_label", type=str)
    parser.add_argument("--input_folder", type=str)
    parser.add_argument("--iter_num", type=int)
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()
    return args


def predict(df, model, class_label, column_name='text_lowercase'):
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
    path = f'{args.input_folder}/{iter_names[args.iter_num]}/train_test/val_{args.class_label}.csv'
    # load the data
    df = pd.read_csv(path)
    # make lower case
    df['text_lowercase'] = df.text.map(str.lower)
    # load model
    model = pickle.load(open(f'models/{iter_names[args.iter_num]}/{args.class_label}.pkl', 'rb'))
    # predict using the loaded model
    preds = predict(df, model, args.class_label)
    # count number of tweet with score higher than threshold
    tweets_gt_threshold = count(preds, args.threshold)
    print(f"We got {tweets_gt_threshold} tweets with 'score > {args.threshold}' out of {len(preds)} for {args.class_label}")
