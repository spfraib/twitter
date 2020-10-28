import pandas as pd
import numpy as np
from time import time
import re
import string
import socket
from glob import glob
import os
import argparse
import pyarrow.parquet as pq


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str,
                        help="Country code",
                        default="US")
    parser.add_argument("--path_to_data", type=str, help="Path to your data.",
                        default=None)

    args = parser.parse_args()
    return args


def is_labeled(x):
    # If First sequence was allocated more than once
    if x[0] > 1:
        # If no other sequence
        if len(x) == 1:
            return True
        else:
            # If second sequence less popular
            if x[1] < x[0]:
                return True
    return False

if __name__ == "__main__":
    args = get_args_from_command_line()
    # Country Code
    country_code = args.country_code
    path_to_data = f'/scratch/mt4493/twitter_labor/twitter-labor-data/data/qualtrics/{args.country_code}/labeling'

    print('Path to data:', path_to_data)
    print('Country Code:', country_code)

    # # Collect all existing labels

    print("Surveys:", len(sorted([x.split('/')[-2] for x in glob(
        os.path.join(path_to_data, 'classification', country_code, 'labeling', 'qualtrics', '*', 'labels.csv'))])))

    # Only keep one label per worker and tweet
    labels = pd.concat(
        [pd.read_csv(file) for file in glob(
            os.path.join(path_to_data, 'classification', country_code, 'labeling', 'qualtrics', '*',
                         'labels.csv'))]).sort_values(
        by=['tweet_id', 'class_id', 'QIDWorker']).drop_duplicates(
        ['tweet_id', 'class_id', 'QIDWorker']).set_index(
        ['tweet_id', 'class_id', 'QIDWorker'])

    print('# labels:', labels.shape[0])

    # Counts labels for each observation
    counts = labels.groupby(['tweet_id', 'class_id'])['score'].value_counts().rename('count')

    # Keep tweets that were labeled more than once with most popular labels strictly dominating
    ids_labeled = counts.groupby(['tweet_id', 'class_id']).apply(list).apply(is_labeled).groupby('tweet_id').sum().where(
        lambda x: x == 5).dropna().index
    print('# labeled tweets:', len(ids_labeled))

    # +
    # Keep most popular label sequence
    labels = counts.reindex(ids_labeled, level='tweet_id').reset_index(
        level='score').groupby(['tweet_id', 'class_id'])['score'].first().unstack()
    labels.index = labels.index.astype(str)

    class2name = dict(zip(range(1, 6), [
        'is_unemployed',
        'lost_job_1mo',
        'job_search',
        'is_hired_1mo',
        'job_offer',
    ]))

    # ['Does this tweet indicate that the user is currently unemployed?',
    # 'Does this tweet indicate that the user became unemployed within the last month?',
    # 'Does this tweet indicate that the user is currently searching for a job?',
    # 'Does this tweet indicate that the user was hired within the last month?',
    # 'Does this tweet contain a job offer?', ]

    labels.rename(columns=lambda x: class2name[x], inplace=True)
    labels.reset_index(inplace=True)
    labels.columns.name = ''

    labels.to_pickle(os.path.join(path_to_data, 'classification', country_code, 'labeling', 'labels.pkl'))

    labels.tail()
