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
    parser.add_argument("--iteration_number", type=str)
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
    path_to_data = f'/scratch/mt4493/twitter_labor/twitter-labor-data/data/qualtrics/{country_code}/iter{args.iteration_number}/labeling'

    print('Path to data:', path_to_data)
    print('Country Code:', country_code)

    # # Collect all existing labels

    print("Surveys:", len(sorted([x.split('/')[-2] for x in glob(
        os.path.join(path_to_data, '*', 'labels.csv'))])))

    # Only keep one label per worker and tweet
    labels = pd.concat(
        [pd.read_csv(file) for file in glob(
            os.path.join(path_to_data, '*',
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

    nb_tweets_to_label_dict = {
        'US': 4081,
        'MX': 2783,
        'BR': 2485
    }
    print('# tweets to label (new sample):', nb_tweets_to_label_dict[args.country_code])
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

    labels.to_parquet(os.path.join(path_to_data, 'labels.parquet'))

    print(labels.tail())

    if args.country_code == 'US' and args.iteration_number == "0":
        list_old_survey_ids = ['SV_7aeWaaTnG7NiYOV',
         'SV_8uMuwiJVgsGDPjn',
         'SV_81Z6plk4o7m4k0R',
         'SV_6mSgd3aLStLTkcl',
         'SV_d6Z7MnJPXSPYnMp',
         'SV_dhDa4Jrlt5rEA7j',
         'SV_b7rYWRVD9CE04dL',
         'SV_9GGdQAjHupi5OKx',
         'SV_eQdmwpIo95tdItD',
         'SV_6JZyroFZpUizsjj',
         'SV_54hSt6qDYsbKAxn',
         'SV_bkhY4hq3qfNnWap',
         'SV_0ClQjqjqerIqZNj',
         'SV_8Dk9huGrMrtFesZ',
         'SV_9TzzayDad3RgIXX',
         'SV_3CrEKpGhqlrO8pT',
         'SV_1FEeJM9n3Pi8Azr',
         'SV_agTJ6PT6XgKqRCd',
         'SV_0dB80s8q5OhAV8x',
         'SV_1X0kskPK25dnt6l',
         'SV_7V75YTlrECFweKV',
         'SV_ctODgZS3rLY5rz7',
         'SV_9FWQ2zw1kp5gXIx',
         'SV_8iXk8NykLgCQEIt',
         'SV_cvxRxMs5UNulogd',
         'SV_er3tETgYDXv1G85',
         'SV_6ydfWA2LrVqOCBD',
         'SV_1MRHgIP6EZSuXWt',
         'SV_9FQXSDKa50C8iKp',
         'SV_3fsqdhfXVkxsaSV',
         'SV_5mv8DI1N0sXTZgp',
         'SV_24RoQ3TAAnEpaN7',
         'SV_7aFs0rDpHMfX4ah',
         'SV_4YjcoEjVDDreyrP',
         'SV_0Ilb9QkeyHziljT']

        labels_path_list = glob(os.path.join(path_to_data, '*','labels.csv'))
        labels_path_list = [labels_path for labels_path in labels_path_list if not any(survey_id in labels_path for survey_id in list_old_survey_ids)]
        # Only keep one label per worker and tweet
        new_labels = pd.concat(
            [pd.read_csv(file) for file in labels_path_list]).sort_values(
            by=['tweet_id', 'class_id', 'QIDWorker']).drop_duplicates(
            ['tweet_id', 'class_id', 'QIDWorker']).set_index(
            ['tweet_id', 'class_id', 'QIDWorker'])

        print('# labels (new sample):', new_labels.shape[0])

        # Counts labels for each observation
        counts = new_labels.groupby(['tweet_id', 'class_id'])['score'].value_counts().rename('count')

        # Keep tweets that were labeled more than once with most popular labels strictly dominating
        ids_labeled = counts.groupby(['tweet_id', 'class_id']).apply(list).apply(is_labeled).groupby(
            'tweet_id').sum().where(
            lambda x: x == 5).dropna().index

        print('# labeled tweets (new sample):', len(ids_labeled))

        # +
        # Keep most popular label sequence
        new_labels = counts.reindex(ids_labeled, level='tweet_id').reset_index(
            level='score').groupby(['tweet_id', 'class_id'])['score'].first().unstack()
        new_labels.index = new_labels.index.astype(str)

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

        new_labels.rename(columns=lambda x: class2name[x], inplace=True)
        new_labels.reset_index(inplace=True)
        new_labels.columns.name = ''

        new_labels.to_pickle(os.path.join(path_to_data, 'new_labels.pkl'))

        new_labels.tail()

