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
import socket
import pyarrow.parquet as pq
import argparse

def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", type=int,
                        help="The number of timelines to download before saving", default=1000)
    parser.add_argument("--country_code", type=str)
    parser.add_argument("--this_batch", type=str, default=None)


    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args_from_command_line()

    # # Params

    cutoff = args.cutoff
    print('Save Data After Downloading',cutoff,'Timelines')

    # +
    country_codes=[
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
    ]

    country_code = args.country_code
    print('Country:', country_code)

    this_batch = args.this_batch
    print('This batch:', this_batch)


    # +
    def get_env_var(varname,default):

        if os.environ.get(varname) != None:
            var = int(os.environ.get(varname))
            print(varname,':', var)
        else:
            var = default
            print(varname,':', var,'(Default)')
        return var

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
    path_to_timelines = os.path.join(path_to_data,'timelines')
    os.makedirs(os.path.join(path_to_timelines,this_batch,'API',country_code), exist_ok=True)
    print(path_to_keys)
    print(path_to_timelines)


    # -

    # # Credentials

    # +
    def get_key_files(SLURM_ARRAY_TASK_ID,SLURM_ARRAY_TASK_COUNT,SLURM_JOB_CPUS_PER_NODE):

        # Randomize set of key files using constant seed
        np.random.seed(0)
        all_key_files = np.random.permutation(glob(os.path.join(path_to_keys,'*.json')))

        # Split file list by node
        key_files = np.array_split(all_key_files,SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID]

        # Check that node has more CPU than key file
        if len(key_files) <= SLURM_JOB_CPUS_PER_NODE:
            print('# Credentials Allocated To Node:', len(key_files))
        else:
            print('Check environment variables:')
            print('# Credentials (',len(key_files),') > # CPU (', SLURM_JOB_CPUS_PER_NODE,')')
            print('Only keeping', SLURM_JOB_CPUS_PER_NODE, 'credentials')
            key_files = key_files[:SLURM_JOB_CPUS_PER_NODE]

        return key_files

    key_files = get_key_files(SLURM_ARRAY_TASK_ID,SLURM_ARRAY_TASK_COUNT,SLURM_JOB_CPUS_PER_NODE)
    print('\n'.join(key_files))


    # +
    def get_auth(key_file):

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
            print(key_file,": Authentication checked")
        except:
            print(key_file,": error during authentication")
            sys.exit('Exit')

        return api

    for key_file in key_files:
        api=get_auth(key_file)
    print('Credentials Checked!')
    # -

    # # User List

    # +
    start = timer()
    print('Select Users...')

    # Select most recent id across pulls
    users=pq.ParquetDataset(glob(os.path.join(path_to_timelines,'*','most_recent_id',country_code,'*.parquet'))).read().to_pandas()

    # Keep the most recent tweets for each user
    users=users.sort_values(['user_id','created_at'],ascending=[True,False]).drop_duplicates('user_id',keep='first')

    # Randomize users
    users=users.sample(frac=1,random_state=0)

    print('# Users :', len(users))

    end = timer()
    print('Computing Time:', round(end - start), 'sec')

    # +
    start = timer()
    print('Split Users Across Nodes...')

    print('First user:', users.index[0])
    users=np.array_split(users,SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID]
    print('# Users for this node:', len(users))
    print('First user for this node:', users.index[0])

    end = timer()
    print('Computing Time:', round(end - start), 'sec')

    # +
    start = timer()
    print('Remove users whose timeline were successfully downloaded...')

    def get_success(country_code):

        if not os.path.exists(os.path.join(path_to_timelines, this_batch, 'API', country_code, 'success')):
            return set()
        else:
            success = set()
            with open(os.path.join(path_to_timelines, this_batch, 'API', country_code, 'success'), 'r', encoding='utf-8') as file:
                for line in file:
                    success.add(line.strip('\n').split('\t')[0])
            return set(success)

    success=get_success(country_code)
    print('# Downloaded timelines:', len(success))

    users=users[-users.user_id.isin(success)].copy()
    print('# Users :', len(users))

    end = timer()
    print('Computing Time:', round(end - start), 'sec')


    # -

    # Nb of verified users in the US = 21,205,171

    # # Get Timelines

    # +
    def get_timeline(user_id,tweet_id,api):

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



    # -

    def download_timelines(index_key):

        # Create Access For Block of Users
        api = get_auth(key_files[index_key])

        # Select Block of Users
        users_block = np.array_split(users,len(key_files))[index_key][['user_id','tweet_id']].values.tolist()

        # Initialize Output File ID
        output_id = str(uuid.uuid4())

        # Initialize DataFrame
        timelines = pd.DataFrame()

        # Initialize Downloaded User List
        downloaded_ids = []
        counter_ids = 0

        for (user_id,tweet_id) in users_block:

            # Try Downloading Timeline
            timeline, error = get_timeline(user_id,tweet_id,api)

            if error!=None:
    #             print(user_id,index_key,error)
                continue

            # Append
            timelines = pd.concat([timelines, timeline],sort=False)
            downloaded_ids.append(user_id)

            # Save after <cutoff> timelines or when reaching last user
            if len(downloaded_ids) == cutoff or user_id == users_block[-1][0]:

                counter_ids += len(downloaded_ids)

                filename = \
                'timelines-'+\
                str(SLURM_JOB_ID)+'-'+\
                str(SLURM_ARRAY_TASK_ID)+'-'+\
                str(index_key)+'-'+\
                str(len(downloaded_ids))+'-'+\
                output_id+'.json.bz2'

                print('Process', index_key, 'downloaded', counter_ids, 'timelines with most recent output file:',
                os.path.join(path_to_timelines,this_batch,'API',country_code,filename))

                # Save as list of dict discarding index
                timelines.to_json(
                os.path.join(path_to_timelines,this_batch,'API',country_code,filename),
                orient='records',
                force_ascii=False,
                date_format=None,
                double_precision=15)

                # Save User Id and File In Which Its Timeline Was Saved
                with open(os.path.join(path_to_timelines,this_batch,'API',country_code,'success'), 'a', encoding='utf-8') as file:
                    for downloaded_id in downloaded_ids:
                        file.write(downloaded_id+'\t'+filename+'\n')

                # Reset Output File ID, Data, and Downloaded Users
                del timelines, downloaded_ids
                output_id = str(uuid.uuid4())
                timelines = pd.DataFrame()
                downloaded_ids = []

        return 0


    print('Extract Timelines...\n')
    with mp.Pool() as pool:
        pool.map(download_timelines, range(len(key_files)))
