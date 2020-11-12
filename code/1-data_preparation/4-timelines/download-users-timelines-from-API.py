from timeit import default_timer as timer
import os
import sys
import uuid
from glob import glob
import json
import tweepy
import numpy as np
import pandas as pd
import multiprocessing as mp
import pyarrow.parquet as pq
import argparse

def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", type=int,
                        help="The number of timelines to download before saving")
    parser.add_argument("--country_code", type=str)
    parser.add_argument("--last_batch", type=str)
    parser.add_argument("--this_batch", type=str, default=None)
    parser.add_argument("--get", type=int, default=0)
    parser.add_argument("--update", type=int, default=0)

    args = parser.parse_args()
    return args


def get_env_var(varname, default):
    if os.environ.get(varname) != None:
        var = int(os.environ.get(varname))
        print(varname, ':', var)
    else:
        var = default
        print(varname, ':', var, '(Default)')
    return var


def get_key_files(SLURM_ARRAY_TASK_ID, SLURM_ARRAY_TASK_COUNT, SLURM_JOB_CPUS_PER_NODE):
    """Get the list of key files from the parallelization parameters."""
    # Randomize set of key files using constant seed
    np.random.seed(0)
    all_key_files = np.random.permutation(glob(os.path.join(path_to_keys, '*.json')))

    # Split file list by node
    key_files = np.array_split(all_key_files, SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID]

    # Check that node has more CPU than key file
    if len(key_files) <= SLURM_JOB_CPUS_PER_NODE:
        print('# Credentials Allocated To Node:', len(key_files))
    else:
        print('Check environment variables:')
        print('# Credentials (', len(key_files), ') > # CPU (', SLURM_JOB_CPUS_PER_NODE, ')')
        print('Only keeping', SLURM_JOB_CPUS_PER_NODE, 'credentials')
        key_files = key_files[:SLURM_JOB_CPUS_PER_NODE]

    return key_files


def get_auth(key_file):
    """Build the API connection from the key files."""
    # Import Key
    with open(key_file) as f:
        key = json.load(f)

    # OAuth process, using the keys and tokens
    auth = tweepy.OAuthHandler(key['consumer_key'], key['consumer_secret'])
    auth.set_access_token(key['access_token'], key['access_token_secret'])

    # Creation of the actual interface, using authentication
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    try:
        api.verify_credentials()
        print(key_file, ": Authentication checked")
    except:
        print(key_file, ": error during authentication")
        sys.exit('Exit')

    return api


def get_success(country_code):
    if not os.path.exists(os.path.join(path_to_timelines, country_code, 'success')):
        return set()
    else:
        success = set()
        with open(os.path.join(path_to_timelines, country_code, 'success'), 'r', encoding='utf-8') as file:
            for line in file:
                success.add(line.strip('\n').split('\t')[0])
        return set(success)


def get_timeline(user_id, tweet_id, api):
    timeline = []
    error = None

    # Collect All Statuses in Timeline
    try:
        cursor = tweepy.Cursor(
            api.user_timeline,
            user_id=user_id,
            since_id=tweet_id,
            count=3200,
            tweet_mode="extended",
            include_rts=True).items()

        for status in cursor:
            timeline.append(status._json)

    except tweepy.error.TweepError as e:
        error = str(e)

    return pd.DataFrame(timeline), error


def get_timelines(index_key, country_code):
    # Create Access For Block of Users
    api = get_auth(key_files[index_key])

    # Select Block of Users
    users_block = np.array_split(users, len(key_files))[index_key]

    # Initialize Output File ID
    output_id = str(uuid.uuid4())

    # Initialize DataFrame
    timelines = pd.DataFrame()

    # Initialize Downloaded User List
    downloaded_ids = []

    for user_index, user_id in enumerate(users_block):

        # Try Downloading Timeline
        timeline, error = get_timeline(user_id, api)

        if error != None:
            #             print(user_id,index_key,error)
            continue

        # Append
        timelines = pd.concat([timelines, timeline], sort=False)
        downloaded_ids.append(user_id)

        # Save after <cutoff> timelines or when reaching last user
        if len(downloaded_ids) == cutoff or user_id == users_block[-1]:

            filename = \
                'timelines-' + \
                str(SLURM_JOB_ID) + '-' + \
                str(SLURM_ARRAY_TASK_ID) + '-' + \
                str(index_key) + '-' + \
                str(len(downloaded_ids)) + '-' + \
                output_id + '.json.bz2'

            print('Process', index_key, 'processed', user_index, 'timelines with latest output file:',
                  os.path.join(path_to_timelines, country_code, filename))

            # Save as list of dict discarding index
            timelines.to_json(
                os.path.join(path_to_timelines, country_code, filename),
                orient='records',
                force_ascii=False,
                date_format=None,
                double_precision=15)

            # Save User Id and File In Which Its Timeline Was Saved
            with open(os.path.join(path_to_timelines, country_code, 'success'), 'a', encoding='utf-8') as file:
                for downloaded_id in downloaded_ids:
                    file.write(downloaded_id + '\t' + filename + '\n')

            # Reset Output File ID, Data, and Downloaded Users
            del timelines, downloaded_ids
            output_id = str(uuid.uuid4())
            timelines = pd.DataFrame()
            downloaded_ids = []

    return 0


def update_timelines(index_key):
    # Create Access For Block of Users
    api = get_auth(key_files[index_key])

    # Select Block of Users
    users_block = np.array_split(users, len(key_files))[index_key][['user_id', 'tweet_id']].values.tolist()

    # Initialize Output File ID
    output_id = str(uuid.uuid4())

    # Initialize DataFrame
    timelines = pd.DataFrame()

    # Initialize Downloaded User List
    downloaded_ids = []
    counter_ids = 0

    for (user_id, tweet_id) in users_block:

        # Try Downloading Timeline
        timeline, error = get_timeline(user_id, tweet_id, api)

        if error != None:
            #             print(user_id,index_key,error)
            continue

        # Append
        timelines = pd.concat([timelines, timeline], sort=False)
        downloaded_ids.append(user_id)

        # Save after <cutoff> timelines or when reaching last user
        if len(downloaded_ids) == cutoff or user_id == users_block[-1][0]:

            counter_ids += len(downloaded_ids)

            filename = \
                'timelines-' + \
                str(SLURM_JOB_ID) + '-' + \
                str(SLURM_ARRAY_TASK_ID) + '-' + \
                str(index_key) + '-' + \
                str(len(downloaded_ids)) + '-' + \
                output_id + '.json.bz2'

            print('Process', index_key, 'downloaded', counter_ids, 'timelines with most recent output file:',
                  os.path.join(path_to_timelines, country_code, filename))

            # Save as list of dict discarding index
            timelines.to_json(
                os.path.join(path_to_timelines, country_code, filename),
                orient='records',
                force_ascii=False,
                date_format=None,
                double_precision=15)

            # Save User Id and File In Which Its Timeline Was Saved
            with open(os.path.join(path_to_timelines, country_code, 'success'), 'a', encoding='utf-8') as file:
                for downloaded_id in downloaded_ids:
                    file.write(downloaded_id + '\t' + filename + '\n')

            # Reset Output File ID, Data, and Downloaded Users
            del timelines, downloaded_ids
            output_id = str(uuid.uuid4())
            timelines = pd.DataFrame()
            downloaded_ids = []

    return 0

if __name__ == "__main__":
    args = get_args_from_command_line()
    # Params
    cutoff = args.cutoff
    country_code = args.country_code
    last_batch = args.last_batch
    this_batch = args.this_batch if args.update == 1 else "historical"

    # Choose Number of Nodes To Distribute Credentials: e.g. jobarray=0-4, cpu_per_task=20, credentials = 90 (<100)
    SLURM_JOB_ID = get_env_var('SLURM_JOB_ID', 0)
    SLURM_ARRAY_TASK_ID = get_env_var('SLURM_ARRAY_TASK_ID', 0)
    SLURM_ARRAY_TASK_COUNT = get_env_var('SLURM_ARRAY_TASK_COUNT', 1)
    SLURM_JOB_CPUS_PER_NODE = get_env_var('SLURM_JOB_CPUS_PER_NODE', mp.cpu_count())

    # Define paths
    path_to_data = '/scratch/spf248/twitter/data'
    path_to_keys = os.path.join(path_to_data, 'keys', 'twitter')
    path_to_timelines = os.path.join(path_to_data, 'timelines', this_batch, 'API')
    if args.get == 1:
        path_to_users = os.path.join(path_to_data, 'users')
        path_to_locations = os.path.join(path_to_data, 'locations', 'profiles')
        os.makedirs(path_to_timelines, exist_ok=True)
        print(f'Path to locations: {path_to_locations}')
    elif args.update == 1:
        path_to_users = os.path.join(path_to_data, 'timelines', last_batch, 'most_recent_id')
        os.makedirs(os.path.join(path_to_timelines, country_code), exist_ok=True)
    print(f'Path to users: {path_to_users}')
    print(f'Path to keys: {path_to_keys}')
    print(f'Path to timelines: {path_to_timelines}')

    # Get key files
    key_files = get_key_files(SLURM_ARRAY_TASK_ID, SLURM_ARRAY_TASK_COUNT, SLURM_JOB_CPUS_PER_NODE)
    print('Key files: ', '\n'.join(key_files))

    if args.get == 1:
        print('Import Users By Account Locations')
        start = timer()

        l = []
        for filename in sorted(glob(os.path.join(path_to_users, 'user-ids-by-account-location-verified/*.json'))):
            try:
                df = pd.read_json(filename, lines=True)
                l.append(df)
            except:
                print('error importing', filename)

        users_by_account_location = pd.concat(l, axis=0, ignore_index=True)
        users_by_account_location = users_by_account_location.set_index('user_location')['user_id']
        users_by_account_location = users_by_account_location.apply(eval).apply(lambda x: [str(y) for y in x])
        print('# Locations:', len(users_by_account_location))
        print('# Users Total:', users_by_account_location.apply(len).sum())

        end = timer()
        print('Computing Time:', round(end - start), 'sec')

        # %%
        print('Import Locations')
        account_locations = pd.read_pickle(os.path.join(path_to_locations, 'account-locations.pkl'))
        print('# Locations:', len(account_locations))

        # %%
        start = timer()
        print('Select Users...')

        # Sorted list of users in selected countries
        users = pd.merge(
            users_by_account_location.reindex(
                account_locations.loc[
                    account_locations['country_short'] == country_code, 'user_location']).dropna().reset_index(),
            account_locations[['user_location', 'country_short']]).drop('user_location', 1).rename(
            columns={
                'country_short': 'country_code'}).explode('user_id').set_index('user_id')['country_code'].sort_index()

        del users_by_account_location
        del account_locations

    elif args.update == 1:
        start = timer()
        print('Select Users...')

        users = pq.ParquetDataset(glob(os.path.join(path_to_users, country_code, '*.parquet'))).read().to_pandas()

    # Randomize users
    users = users.sample(frac=1, random_state=0)

    print('# Users :', len(users))

    end = timer()
    print('Computing Time:', round(end - start), 'sec')

    start = timer()
    print('Split Users Across Nodes...')

    print('First user:', users.index[0])
    users = np.array_split(users, SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID]
    print('# Users for this node:', len(users))
    print('First user for this node:', users.index[0])

    end = timer()
    print('Computing Time:', round(end - start), 'sec')

    start = timer()
    print('Remove users whose timeline were successfully downloaded...')

    if args.get == 1:
        success = set()

        tmp = get_success(country_code)
        print(country_code, ':', len(tmp))
        success = success.union(tmp)
        print('# Downloaded timelines:', len(success))

        users.drop(success, errors='ignore', inplace=True)
        print('# Remaining users for this node:', len(users))

        #users_by_country = users.reset_index().groupby('country_code')['user_id'].apply(list).reindex(country_codes)

    elif args.update == 1:
        success = get_success(country_code)
        print('# Downloaded timelines:', len(success))

        users = users[-users.user_id.isin(success)].copy()
        print('# Users :', len(users))

    end = timer()
    print('Computing Time:', round(end - start), 'sec')

    print('Extract Timelines...\n')
    if args.get == 1:
        with mp.Pool() as pool:
            pool.map(get_timelines, range(len(key_files)))
    elif args.update == 1:
        with mp.Pool() as pool:
            pool.map(update_timelines, range(len(key_files)))