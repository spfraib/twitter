from timeit import default_timer as timer
import itertools
import os
import sys
import uuid
from glob import glob
import json
import tweepy
import numpy as np
import pandas as pd
import multiprocessing as mp
import psutil
import socket
from functools import partial

# # Params
cutoff = 500
print('Save Data After Downloading',cutoff,'Timelines')

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

# %%
#country_codes=[
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
#]

#TODO: country_codes to argument
country_codes=[
'FR',
]

#TODO: modify path_to_data (copy sam's files to my scratch)
path_to_data='/scratch/spf248/twitter/data'
path_to_users = os.path.join(path_to_data,'users')
path_to_locations = os.path.join(path_to_data,'locations','profiles')
path_to_keys = os.path.join(path_to_data,'keys','twitter')
path_to_timelines = os.path.join(path_to_data,'timelines','historical','API')
os.makedirs(path_to_timelines, exist_ok=True)
print(path_to_users)
print(path_to_locations)
print(path_to_keys)
print(path_to_timelines)


# # Credentials

#TODO: how come we have several key files?
def get_key_files(SLURM_ARRAY_TASK_ID,SLURM_ARRAY_TASK_COUNT,SLURM_JOB_CPUS_PER_NODE):
    """
    A partir des parametres de la parallelisation, donne liste de key files
    :param SLURM_ARRAY_TASK_ID:
    :param SLURM_ARRAY_TASK_COUNT:
    :param SLURM_JOB_CPUS_PER_NODE:
    :return:
    """
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


def get_auth(key_file):
    """
    A partir de key_file, crée connexion a l'API
    :param key_file:
    :return:
    """
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

# for key_file in np.random.permutation(glob(os.path.join(path_to_keys,'*.json'))):
#     get_auth(key_file)
# print('Credentials Checked!')


# %% [markdown]
# # User List

# %%
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
print('# Users Total:', users_by_account_location.apply(len).sum())

end = timer()
print('Computing Time:', round(end - start), 'sec')

# %%
print('Import Locations')
account_locations=pd.read_pickle(os.path.join(path_to_locations,'account-locations.pkl')) 
print('# Locations:', len(account_locations))

# %%
start = timer()
print('Select Users...')

# Sorted list of users in selected countries
users=pd.merge(
users_by_account_location.reindex(
account_locations.loc[
account_locations['country_short'].isin(country_codes),'user_location']).dropna().reset_index(),
account_locations[['user_location','country_short']]).drop('user_location',1).rename(
columns={'country_short':'country_code'}).explode('user_id').set_index('user_id')['country_code'].sort_index()

# Randomize users
users=users.sample(frac=1,random_state=0)

del users_by_account_location
del account_locations

print('# Users :', len(users)) 
print(users.reset_index().groupby('country_code').count())

end = timer()
print('Computing Time:', round(end - start), 'sec')

# %%
start = timer()
print('Split Users Across Nodes...')

print('First user:', users.index[0])
users=np.array_split(users,SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID]
print('# Users for this node:', len(users)) 
print('First user for this node:', users.index[0])

end = timer()
print('Computing Time:', round(end - start), 'sec')

# %% [markdown]
# Nb of verified users in the US = 21,205,171

# %%
start = timer()
print('Remove users whose timeline were successfully downloaded...')

# previously downloaded?

def get_success(country_code):
    
    if not os.path.exists(os.path.join(path_to_timelines, country_code, 'success')):
        return set()
    else:
        success = set()
        with open(os.path.join(path_to_timelines, country_code, 'success'), 'r', encoding='utf-8') as file:
            for line in file:
                success.add(line.strip('\n').split('\t')[0])
        return set(success)

success=set()
for country_code in country_codes:
    tmp=get_success(country_code)
    print(country_code, ':', len(tmp))
    success=success.union(tmp)
print('# downloaded timelines:', len(success))

users.drop(success,errors='ignore',inplace=True)
print('# remaining users for this node:', len(users))

# Group users by country
users_by_country=users.reset_index().groupby('country_code')['user_id'].apply(list).reindex(country_codes)

end = timer()
print('Computing Time:', round(end - start), 'sec')


# %% [markdown]
# # Get Timelines

# %%
def get_timeline(user_id,api):
    
    timeline = []
    error = None
    
    # Collect All Statuses in Timeline
    try:
        cursor = tweepy.Cursor(
        api.user_timeline, 
        user_id=user_id, 
        count=3200,
        tweet_mode="extended", 
        include_rts=True).items()
        
        for status in cursor:
            timeline.append(status._json)
     
    except tweepy.error.TweepError as e:
        error = str(e)
        
    return pd.DataFrame(timeline), error

# timeline = get_user_timeline('12',get_auth(key_file))


# %%
def download_timelines(index_key,country_code):

    # Create Access For Block of Users
    api = get_auth(key_files[index_key])
    
    # Select Block of Users
    users_block = np.array_split(users_by_country[country_code],len(key_files))[index_key]
    
    # Initialize Output File ID
    output_id = str(uuid.uuid4())
    
    # Initialize DataFrame
    timelines = pd.DataFrame()
    
    # Initialize Downloaded User List
    downloaded_ids = []
    
    for user_index,user_id in enumerate(users_block):
        
        # Try Downloading Timeline
        timeline, error = get_timeline(user_id,api)
        
        if error!=None:
#             print(user_id,index_key,error)
            continue
            
        # Append
        timelines = pd.concat([timelines, timeline],sort=False)
        downloaded_ids.append(user_id)
            
        # Save after <cutoff> timelines or when reaching last user
        if len(downloaded_ids) == cutoff or user_id == users_block[-1]:
            
            filename = \
            'timelines-'+\
            str(SLURM_JOB_ID)+'-'+\
            str(SLURM_ARRAY_TASK_ID)+'-'+\
            str(index_key)+'-'+\
            str(len(downloaded_ids))+'-'+\
            output_id+'.json.bz2'
            
            print('Process', index_key, 'processed', user_index, 'timelines with latest output file:', 
            os.path.join(path_to_timelines,country_code,filename))
            
            # Save as list of dict discarding index
            timelines.to_json(
            os.path.join(path_to_timelines,country_code,filename),
            orient='records',
            force_ascii=False,
            date_format=None,
            double_precision=15)
            
            # Save User Id and File In Which Its Timeline Was Saved
            with open(os.path.join(path_to_timelines,country_code,'success'), 'a', encoding='utf-8') as file:
                for downloaded_id in downloaded_ids:
                    file.write(downloaded_id+'\t'+filename+'\n')
            
            # Reset Output File ID, Data, and Downloaded Users
            del timelines, downloaded_ids
            output_id = str(uuid.uuid4())
            timelines = pd.DataFrame()
            downloaded_ids = []
            
    return 0


# %%
print('Extract Timelines...\n')
with mp.Pool() as pool:
    for country_code in country_codes:
        print(country_code)
        pool.map(partial(download_timelines, country_code=country_code), range(len(key_files)))
