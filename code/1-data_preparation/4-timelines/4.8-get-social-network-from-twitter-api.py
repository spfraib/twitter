#!/usr/bin/env python
# coding: utf-8

# http://holoviews.org/user_guide/Network_Graphs.html
#     
# https://pyvis.readthedocs.io/en/latest/tutorial.html
#     
# https://towardsdatascience.com/python-interactive-network-visualization-using-networkx-plotly-and-dash-e44749161ed7
#     
# https://github.com/tweepy/tweepy/issues/627
#     
# https://blog.f-secure.com/how-to-get-twitter-follower-data-using-python-and-tweepy/
#     
# https://stackoverflow.com/questions/31000178/how-to-get-large-list-of-followers-tweepy

# In[1]:


from timeit import default_timer as timer
import os
import socket
from glob import glob
import json
import tweepy
import numpy as np
import pandas as pd
import multiprocessing as mp
import argparse
from utils import get_env_var, get_key_files, get_auth


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", type=int,
                        help="The number of timelines to download before saving", default=1000)
    parser.add_argument("--country_code", type=str)
    parser.add_argument("--this_batch", type=str, default=None)

    args = parser.parse_args()
    return args


def select_users(users_all, SLURM_ARRAY_TASK_ID, SLURM_ARRAY_TASK_COUNT):
    # Randomize All Users
    np.random.seed(0)
    users_all = np.random.permutation(users_all)

    print('# Users:', len(users_all))
    print('First User:', users_all[0])

    # Split users by node
    users_selected = np.array_split(users_all, SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID].copy()
    print('Node"s # Users:', len(users_selected))
    print('Node"s First User:', users_selected[0])

    return set(users_selected)


def user_timeline(api, user_id):
    timeline = []

    try:
        cursor = tweepy.Cursor(
            api.user_timeline,
            user_id=user_id,
            count=3200,
            tweet_mode="extended",
            include_rts=True).items()
        for status in cursor:
            timeline.append(status._json)
        return timeline

    except tweepy.error.TweepError as e:
        print(e)
        with open(os.path.join(path_to_data, 'error.txt'), 'a', encoding='utf-8') as file:
            file.write(user_id + '\tuser_timeline\terror\t' + str(e) + '\n')


def friends_ids(api, user_id):
    friends = []

    try:
        cursor = tweepy.Cursor(api.friends_ids, user_id=user_id, count=5000).items()
        for friend in cursor:
            friends.append(friend)
        return friends

    except tweepy.error.TweepError as e:
        print(e)
        with open(os.path.join(path_to_data, 'error.txt'), 'a', encoding='utf-8') as file:
            file.write(user_id + '\tfriends_ids\terror\t' + str(e) + '\n')


def followers_ids(api, user_id):
    followers = []

    try:
        cursor = tweepy.Cursor(api.followers_ids, user_id=user_id, count=5000).items()
        for follower in cursor:
            followers.append(follower)
        return followers

    except tweepy.error.TweepError as e:
        print(e)
        with open(os.path.join(path_to_data, 'error.txt'), 'a', encoding='utf-8') as file:
            file.write(user_id + '\tfollowers_ids\terror\t' + str(e) + '\n')


def get_data_by_block(index_key):
    # Create Access For Block of Users
    api = get_auth(key_files[index_key])

    # Select Block of Users
    users_block = np.array_split(users_selected, len(key_files))[index_key]

    os.makedirs(os.path.join(path_to_data, 'timelines', 'level-' + str(level)), exist_ok=True)
    os.makedirs(os.path.join(path_to_data, 'friends', 'level-' + str(level)), exist_ok=True)
    os.makedirs(os.path.join(path_to_data, 'followers', 'level-' + str(level)), exist_ok=True)

    for i, user_id in enumerate(users_block):

        timeline = user_timeline(api, user_id)
        friends = friends_ids(api, user_id)
        followers = followers_ids(api, user_id)

        if timeline != None and friends != None and followers != None:
            with open(os.path.join(path_to_data, 'timelines', 'level-' + str(level), 'timeline-' + user_id + '.json'),
                      'w') as f:
                json.dump(timeline, f)
            with open(os.path.join(path_to_data, 'friends', 'level-' + str(level), 'friends-' + user_id + '.txt'),
                      'w') as f:
                f.write("\n".join([str(friend) for friend in friends]))
            with open(os.path.join(path_to_data, 'followers', 'level-' + str(level), 'followers-' + user_id + '.txt'),
                      'w') as f:
                f.write("\n".join([str(follower) for follower in followers]))
            with open(os.path.join(path_to_data, 'existing.txt'), 'a', encoding='utf-8') as file:
                file.write(user_id + '\n')


if __name__ == '__main__':
    # # Params
    # Choose Number of Nodes To Distribute Credentials: e.g. jobarray=0-4, cpu_per_task=20, credentials = 90 (<100)
    SLURM_JOB_ID = get_env_var('SLURM_JOB_ID', 0)
    SLURM_ARRAY_TASK_ID = get_env_var('SLURM_ARRAY_TASK_ID', 0)
    SLURM_ARRAY_TASK_COUNT = get_env_var('SLURM_ARRAY_TASK_COUNT', 1)
    SLURM_JOB_CPUS_PER_NODE = get_env_var('SLURM_JOB_CPUS_PER_NODE', mp.cpu_count())

    # # Local
    # if 'samuel' in socket.gethostname().lower():
    #     path_to_data = '../data'
    #     path_to_keys = '../keys'
    # # Cluster
    # else:
    #     path_to_data = '/scratch/spf248/radicom/data'
    #     path_to_keys = '/scratch/spf248/radicom/keys'
    path_to_data = '/scratch/spf248/twitter/data'
    path_to_keys = os.path.join(path_to_data, 'keys', 'twitter')
    print(path_to_data)
    print(path_to_keys)

    level = 1
    print('Level', level)

    # # Credentials

    key_files = get_key_files(SLURM_ARRAY_TASK_ID, SLURM_ARRAY_TASK_COUNT, SLURM_JOB_CPUS_PER_NODE, path_to_keys)
    print('\n'.join(key_files))

    for key_file in np.random.permutation(glob(os.path.join(path_to_keys, '*.json'))):
        get_auth(key_file)
    print('Credentials Checked!')

    # # Users List

    print('Import Friends and Followers List')
    start = timer()

    users_all = []
    # Take previous level as input
    for filename in sorted(glob(os.path.join(path_to_data, 'friends/level-' + str(level - 1) + '/*.txt')) + glob(
            os.path.join(path_to_data, 'followers/level-' + str(level - 1) + '/*.txt'))):

        with open(filename, 'r') as f:
            for line in f:
                users_all.append(line.strip('\n'))
    users_all = list(set(users_all))

    end = timer()
    print('Computing Time:', round(end - start), 'sec')

    start = timer()
    print('Select Users...')

    users_selected = select_users(users_all, SLURM_ARRAY_TASK_ID, SLURM_ARRAY_TASK_COUNT)

    end = timer()
    print('Computing Time:', round(end - start), 'sec')

    print('Remove Existing Users:')
    if os.path.exists(os.path.join(path_to_data, 'existing.txt')):
        users_existing = set(
            pd.read_csv(os.path.join(path_to_data, 'existing.txt'), header=None, squeeze=True, dtype='str'))
        users_selected = users_selected.difference(users_existing)

    np.random.seed(0)
    users_selected = np.random.permutation(list(users_selected))
    print('# Remaining Users:', len(users_selected))

    # # Download

    start = timer()
    print('Extract Data By Block...\n')

    with mp.Pool() as pool:

        pool.map(get_data_by_block, range(len(key_files)))

    end = timer()
    print('Computing Time:', round(end - start), 'sec')
