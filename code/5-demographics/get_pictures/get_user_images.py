import os
from timeit import default_timer as timer
from glob import glob
import pandas as pd
import pyarrow.parquet as pq
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import wget
import multiprocessing as mp
import argparse


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str,
                        default="US")
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


if __name__ == '__main__':
    args = get_args_from_command_line()

    country_code = args.country_code
    print('Country:', country_code)

    # Choose Number of Nodes To Distribute Credentials: e.g. jobarray=0-4, cpu_per_task=20, credentials = 90 (<100)
    SLURM_JOB_ID = get_env_var('SLURM_JOB_ID', 0)
    SLURM_ARRAY_TASK_ID = get_env_var('SLURM_ARRAY_TASK_ID', 0)
    SLURM_ARRAY_TASK_COUNT = get_env_var('SLURM_ARRAY_TASK_COUNT', 1)
    SLURM_JOB_CPUS_PER_NODE = get_env_var('SLURM_JOB_CPUS_PER_NODE', mp.cpu_count())

    if os.getenv('CLUSTER') == 'PRINCE':
        path_to_data = '/scratch/spf248/twitter/data'
    else:
        path_to_data = '../../data'

    output_dir = f"/scratch/spf248/twitter/data/classification/{country_code}/profile_pictures_sam/{str(SLURM_ARRAY_TASK_ID)}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    error_log = open(os.path.join(output_dir, f"erroneous_users_{SLURM_ARRAY_TASK_ID}"), 'w')

    print(path_to_data)
    print('Load')
    start = timer()
    dir_name = f"/scratch/spf248/twitter/data/classification/{args.country_code}/users/"
    # filenames = os.listdir("/scratch/spf248/twitter/data/classification/US/users/")
    paths_to_filtered = list(
        np.array_split(glob(os.path.join(dir_name, '*.parquet')), SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID])
    print('#files:', len(paths_to_filtered))
    count = 0
    for filename in paths_to_filtered:
        if not filename.endswith("parquet"):
            continue
        # filename = os.path.join(dir_name,filename)
        start = timer()
        users = pq.read_table(filename, columns=['user_id', 'profile_image_url_https']).to_pandas()
        for row in users.itertuples(index=False):
            user_id = row[0]
            url = row[1]
            url = url.replace("_normal", "_400x400")
            format = url.split(".")[-1]
            output_path = os.path.join(output_dir, f"{user_id}.{format}")
            try:
                wget.download(url, output_path)
            except:
                # print("Error: "+user_id)
                error_log.write(user_id + "\t" + url + "\n")
            count += 1
        if count % 1000 == 0:
            print(count)
        print("Done in", round(timer() - start), "sec")
