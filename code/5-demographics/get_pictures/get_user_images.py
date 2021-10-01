import os
from timeit import default_timer as timer
from glob import glob
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import wget
import multiprocessing as mp
import argparse
import logging
import tarfile
from pathlib import Path

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


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
        logging.info(varname, ':', var)
    else:
        var = default
        logging.info(varname, ':', var, '(Default)')
    return var


def tar_output_folder(folder_path, output_path):
    with tarfile.open(output_path, 'w') as tar:
        for file_path in Path(folder_path).glob('*.*'):
            tar.add(file_path)


if __name__ == '__main__':
    args = get_args_from_command_line()

    country_code = args.country_code
    logging.info('Country:', country_code)

    # Choose Number of Nodes To Distribute Credentials: e.g. jobarray=0-4, cpu_per_task=20, credentials = 90 (<100)
    SLURM_JOB_ID = get_env_var('SLURM_JOB_ID', 0)
    SLURM_ARRAY_TASK_ID = get_env_var('SLURM_ARRAY_TASK_ID', 0)
    SLURM_ARRAY_TASK_COUNT = get_env_var('SLURM_ARRAY_TASK_COUNT', 1)
    SLURM_JOB_CPUS_PER_NODE = get_env_var('SLURM_JOB_CPUS_PER_NODE', mp.cpu_count())

    output_dir = f"/scratch/spf248/twitter/data/demographics/profile_pictures/{country_code}/{str(SLURM_ARRAY_TASK_ID)}"
    err_dir = f"/scratch/spf248/twitter/data/demographics/profile_pictures/{country_code}/err"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    error_log = open(os.path.join(err_dir, f"erroneous_users_{SLURM_ARRAY_TASK_ID}"), 'w')

    logging.info('Load')
    data_path = '/scratch/spf248/twitter/data'
    # get user id list
    start = timer()
    dir_name = f'{data_path}/user_timeline/user_timeline_crawled/{args.country_code}'
    # filenames = os.listdir("/scratch/spf248/twitter/data/classification/US/users/")
    paths_to_filtered = list(
        np.array_split(glob(os.path.join(dir_name, '*.parquet')), SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID])
    logging.info('#files:', len(paths_to_filtered))
    count = 0
    for file_path in paths_to_filtered:
        if not file_path.endswith("parquet"):
            continue
        start = timer()
        users = pq.read_table(file_path, columns=['user_id', 'profile_image_url_https']).to_pandas()
        for row in users.itertuples(index=False):
            user_id = row[0]
            url = row[1]
            filename = url.rsplit('/', 1)[-1]
            format = os.path.splitext(filename)[1]
            if '_' in filename:
                new_filename = f'{filename.split("_")[0]}{format}'
                url = url.replace(filename, new_filename)
            output_path = os.path.join(output_dir, f"{user_id}{format}")
            try:
                wget.download(url, output_path)
            except:
                # logging.info("Error: "+user_id)
                error_log.write(user_id + "\t" + url + "\n")
            count += 1
        if count % 1000 == 0:
            logging.info(count)
        logging.info("Done in", round(timer() - start), "sec")
