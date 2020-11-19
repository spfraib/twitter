
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

from timeit import default_timer as timer
import os
import sys
import socket
import uuid
from glob import glob
import json
import tweepy
import numpy as np
import pandas as pd
import multiprocessing as mp
import argparse

import sys
sys.path.append('..')
from utils import get_env_var, get_key_files, get_auth

def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", type=int,
                        help="The number of timelines to download before saving", default=10000)
    parser.add_argument("--country_code", type=str)

    args = parser.parse_args()
    return args


def friends_ids(api, user_id, path_to_friends):
    friends = []

    try:
        cursor = tweepy.Cursor(api.friends_ids, user_id=user_id, count=5000).items()
        for friend in cursor:
            friends.append(friend)
        return friends

    except tweepy.error.TweepError as e:
        print(e)
        with open(os.path.join(path_to_friends, 'errors'), 'a', encoding='utf-8') as file:
            file.write(user_id + '\tfriends_ids\terror\t' + str(e) + '\n')


def get_data_by_block(index_key):
    # Create Access For Block of Users
    api = get_auth(key_files[index_key])

    # Select Block of Users
    users_block = np.array_split(users, len(key_files))[index_key]

    # Initialize Output File ID
    output_id = str(uuid.uuid4())

    # Initialize DataFrame
    users_friends = pd.DataFrame()

    # Initialize Downloaded User List
    downloaded_ids = []
    counter_ids = 0

    for i, user_id in enumerate(users_block):

        # Try Downloading Friends
        friends = friends_ids(api, user_id, path_to_friends)

        if friends == None:
            print('Error:', user_id)
            continue

        # Append
        users_friends = pd.concat([users_friends, pd.DataFrame([(user_id, friends)], columns=['user_id', 'friends'])],
                                  sort=False)
        downloaded_ids.append(user_id)

        # Save after <cutoff> timelines or when reaching last user
        if len(downloaded_ids) == cutoff or user_id == users_block[-1][0]:

            counter_ids += len(downloaded_ids)

            filename = \
                'friends-' + \
                str(SLURM_JOB_ID) + '-' + \
                str(SLURM_ARRAY_TASK_ID) + '-' + \
                str(index_key) + '-' + \
                str(len(downloaded_ids)) + '-' + \
                output_id + '.json.bz2'

            print('Process', index_key, 'downloaded', counter_ids, 'friends list with most recent output file:',
                  os.path.join(path_to_friends, filename))

            # Save as list of dict discarding index
            users_friends.to_json(os.path.join(path_to_friends, filename), orient='records')

            # Save User Id and File In Which Its Timeline Was Saved
            with open(os.path.join(path_to_friends, 'success'), 'a', encoding='utf-8') as file:
                for downloaded_id in downloaded_ids:
                    file.write(downloaded_id + '\t' + filename + '\n')

            # Reset Output File ID, Data, and Downloaded Users
            del users_friends, downloaded_ids
            output_id = str(uuid.uuid4())
            users_friends = pd.DataFrame()
            downloaded_ids = []

    return 0

if __name__ == '__main__':
    # # Params
    args = get_args_from_command_line()

    cutoff = args.cutoff
    print('Save Data After Downloading',cutoff,'Timelines')

    # # +
    # country_codes=[
    # 'US',
    # 'ID',
    # 'BR',
    # 'TR',
    # 'MX',
    # 'AR',
    # 'PH',
    # 'CO',
    # 'MY',
    # 'VE',
    # 'TH',
    # 'PK',
    # ]

    country_code = args.country_code
    print('Country:', country_code)



    # Choose Number of Nodes To Distribute Credentials: e.g. jobarray=0-4, cpu_per_task=20, credentials = 90 (<100)
    SLURM_JOB_ID            = get_env_var('SLURM_JOB_ID',0)
    SLURM_ARRAY_TASK_ID     = get_env_var('SLURM_ARRAY_TASK_ID',0)
    SLURM_ARRAY_TASK_COUNT  = get_env_var('SLURM_ARRAY_TASK_COUNT',1)
    SLURM_JOB_CPUS_PER_NODE = get_env_var('SLURM_JOB_CPUS_PER_NODE',mp.cpu_count())

    # +
    if 'samuel' in socket.gethostname().lower():
        path_to_data='../../data'
    else:
        path_to_data='/scratch/spf248/twitter/data'

    path_to_keys = os.path.join(path_to_data,'keys','twitter')
    path_to_users = os.path.join(path_to_data,'users')
    path_to_locations = os.path.join(path_to_data,'locations','profiles')
    path_to_friends = os.path.join(path_to_data,'friends','API',country_code)
    os.makedirs(path_to_friends, exist_ok=True)
    print(path_to_keys)
    print(path_to_users)
    print(path_to_locations)
    print(path_to_friends)

    # # Credentials
    key_files = get_key_files(SLURM_ARRAY_TASK_ID,SLURM_ARRAY_TASK_COUNT,SLURM_JOB_CPUS_PER_NODE, path_to_keys)
    print('\n'.join(key_files))



    for key_file in np.random.permutation(glob(os.path.join(path_to_keys,'*.json'))):
        get_auth(key_file)
    print('Credentials Checked!')

    # # Users List

    print('Import Users By Account Locations')
    start = timer()

    l = []
    for filename in sorted(glob(os.path.join(path_to_users,'user-ids-by-account-location-verified/*.json'))):
        try:
            df = pd.read_json(filename,lines=True)
            l.append(df)
        except:
            print('error importing', filename)
        
    users_by_account_location=pd.concat(l, axis=0, ignore_index=True)
    users_by_account_location=users_by_account_location.set_index('user_location')['user_id']
    users_by_account_location=users_by_account_location.apply(eval).apply(lambda x:[str(y) for y in x])
    print('# Locations:', len(users_by_account_location))
    print('# Users:', users_by_account_location.apply(len).sum())

    end = timer()
    print('Computing Time:', round(end - start), 'sec')

    users_by_account_location.head()

    print('Import Locations')
    user_locations=pd.read_csv(os.path.join(path_to_locations,'user_locations_geocoded.csv'),index_col=0)
    print('# Locations:', len(user_locations))

    user_locations.head()

    # +
    print('Select Users...')
    start = timer()

    # Sorted list of users in selected countries
    users=users_by_account_location.reindex(user_locations.loc[user_locations['country_short']==country_code,'user_location']).dropna().explode().reset_index(drop=True)

    # Randomize users
    users=users.sample(frac=1,random_state=0)

    del users_by_account_location
    del user_locations

    print('# Users :', len(users))
    print('First user:', users.index[0])

    end = timer()
    print('Computing Time:', round(end - start), 'sec')

    # +
    print('Split Users Across Nodes...')
    start = timer()

    users=np.array_split(users,SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID]
    print('# Users for this node:', len(users))
    print('First user for this node:', users.index[0])

    end = timer()
    print('Computing Time:', round(end - start), 'sec')

    # +
    print('Remove Existing Users:')
    start = timer()

    if os.path.exists(os.path.join(path_to_friends,'success')):
        existing_users=set(pd.read_csv(os.path.join(path_to_friends,'success'),names=['user_id','filename'],dtype='str',sep='\t')['user_id'])
        users=set(users).difference(existing_users)

    np.random.seed(0)
    users=np.random.permutation(list(users))
    print('# Remaining Users:', len(users))

    end = timer()
    print('Computing Time:', round(end - start), 'sec')


    # # Download
    #friends = friends_ids(get_auth(key_file),user_id=12, path_to_friends=path_to_friends)

    print('Extract Data By Block...\n')
    start = timer()

    with mp.Pool() as pool:

        pool.map(get_data_by_block, range(len(key_files)))

    end = timer()
    print('Computing Time:', round(end - start), 'sec')
