import os
import numpy as np
from glob import glob
import json
import tweepy
import sys


def get_env_var(varname, default):
    if os.environ.get(varname) != None:
        var = int(os.environ.get(varname))
        print(varname, ':', var)
    else:
        var = default
        print(varname, ':', var, '(Default)')
    return var


def get_key_files(SLURM_ARRAY_TASK_ID, SLURM_ARRAY_TASK_COUNT, SLURM_JOB_CPUS_PER_NODE, path_to_keys):
    # Randomize set of key files using constant seed
    np.random.seed(0)
    all_key_files = np.random.permutation(glob(os.path.join(path_to_keys, '*.json')))

    # Split file list by node
    key_files = np.array_split(all_key_files, SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID]

    # Check that node has more CPU than key file
    if len(key_files) <= SLURM_JOB_CPUS_PER_NODE:
        print('# Credentials Allocated To Node:', len(key_files))
    else:
        print('# Credentials (', len(key_files), ') > # CPU (', SLURM_JOB_CPUS_PER_NODE, ')')
        print('Only Keeping', SLURM_JOB_CPUS_PER_NODE, 'Credentials')
        key_files = key_files[:SLURM_JOB_CPUS_PER_NODE]

    return key_files

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
        print(key_file, ": Authentication checked")
    except:
        print(key_file, ": error during authentication")
        sys.exit('Exit')

    return api

