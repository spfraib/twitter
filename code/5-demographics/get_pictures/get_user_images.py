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
import shutil
import json

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--country_code", type=str)
    args = parser.parse_args()
    return args


def get_env_var(varname, default):
    if os.environ.get(varname) != None:
        var = int(os.environ.get(varname))
        logger.info(f'{varname}: {var}')
    else:
        var = default
        logger.info(f'{varname}: {var}, (Default)')
    return var


def tar_output_folder(folder_path, output_path):
    with tarfile.open(output_path, 'w') as tar:
        for file_path in Path(folder_path).glob('*.*'):
            tar.add(file_path)


if __name__ == '__main__':
    args = get_args_from_command_line()

    # Choose Number of Nodes To Distribute Credentials: e.g. jobarray=0-4, cpu_per_task=20, credentials = 90 (<100)
    SLURM_JOB_ID = get_env_var('SLURM_JOB_ID', 0)
    SLURM_ARRAY_TASK_ID = get_env_var('SLURM_ARRAY_TASK_ID', 0)
    SLURM_ARRAY_TASK_COUNT = get_env_var('SLURM_ARRAY_TASK_COUNT', 1)
    SLURM_JOB_CPUS_PER_NODE = get_env_var('SLURM_JOB_CPUS_PER_NODE', mp.cpu_count())

    data_path = '/scratch/spf248/twitter_data_collection/data'
    output_dir = os.path.join(data_path, f"profile_pictures/{args.country_code}/tars/{str(SLURM_JOB_ID)}")
    success_log_dir = os.path.join(data_path, f"profile_pictures/{args.country_code}/tars/success")
    err_log_dir = os.path.join(data_path, f"profile_pictures/{args.country_code}/tars/err")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(err_log_dir):
        os.makedirs(err_log_dir, exist_ok=True)
    error_log = open(os.path.join(err_log_dir, f"erroneous_users_{SLURM_JOB_ID}.txt"), 'w')
    # if not os.path.exists(success_log_dir):
    #     os.makedirs(success_log_dir)
    # success_log = open(os.path.join(success_log_dir, f"erroneous_users_{SLURM_JOB_ID}.txt"), 'w')
    logger.info('Load data')
    # get user id list
    start = timer()
    if args.country_code == 'NG':
        dir_name = f'/scratch/spf248/twitter_social_cohesion/data/preprocessed_from_twitter_api/profiles/NG'
    if args.mode == 'from_scratch':
        paths_to_filtered = list(
            np.array_split(glob(os.path.join(dir_name, '*.parquet')), SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID])
        logger.info(f'#files: {len(paths_to_filtered)}')
        count = 0
        os.chdir(output_dir)
        with tarfile.open(f'{output_dir}.tar', 'w') as tar:
            for file_path in paths_to_filtered:
                if not file_path.endswith("parquet"):
                    continue
                start = timer()
                users = pq.read_table(file_path, columns=['user_id', 'user_profile_image_url_https']).to_pandas()
                for row in users.itertuples(index=False):
                    user_id = row[0]
                    url = row[1]
                    filename = url.rsplit('/', 1)[-1]
                    ext = os.path.splitext(filename)[1]
                    if '_' in filename:
                        new_filename = f'{filename.split("_")[0]}{ext}'
                        url = url.replace(filename, new_filename)
                    output_path = os.path.join(output_dir, f"{user_id}{ext}")
                    try:
                        wget.download(url, output_path)
                        # success_log.write(f'{user_id}\t{url}\n')
                    except:
                        error_log.write(f'{user_id}\t{url}\n')
                    count += 1
                    if count % 1000 == 0:
                        logger.info(f'Covered {count} users')
                        for f in os.listdir(output_dir):
                            basename = os.path.basename(os.path.join(output_dir, f))
                            tar.add(os.path.join(output_dir, f), arcname=basename)
                            os.remove(os.path.join(output_dir, f))
                logger.info(f"Done in {round(timer() - start)} sec")
            # add last downloaded pictures
            for f in os.listdir(output_dir):
                basename = os.path.basename(os.path.join(output_dir, f))
                tar.add(os.path.join(output_dir, f), arcname=basename)
                os.remove(os.path.join(output_dir, f))
            # remove empty directory
            if os.path.exists(output_dir):
                os.rmdir(output_dir)
    elif args.mode == 'get_missing':
        with open(f'{data_path}/profile_pictures/{args.country_code}/tars/user_ids_w_missing_pics_all.json',
                  'r') as f:
            missing_pics_ids_dict = json.load(f)
        ids_to_collect_list = list(np.array_split(list(missing_pics_ids_dict.keys()), SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID])
        logger.info(f'#pics to collect: {len(ids_to_collect_list)}')
        count = 0
        os.chdir(output_dir)
        with tarfile.open(f'{output_dir}.tar', 'w') as tar:
            for user_id in ids_to_collect_list:
                url = missing_pics_ids_dict[user_id]
                filename = url.rsplit('/', 1)[-1]
                ext = os.path.splitext(filename)[1]
                if '_' in filename:
                    new_filename = f'{filename.split("_")[0]}{ext}'
                    url = url.replace(filename, new_filename)
                output_path = os.path.join(output_dir, f"{user_id}{ext}")
                try:
                    wget.download(url, output_path)
                    # success_log.write(f'{user_id}\t{url}\n')
                except:
                    error_log.write(f'{user_id}\t{url}\n')
                count += 1
                if count % 100 == 0:
                    logger.info(f'Covered {count} users')
                    for f in os.listdir(output_dir):
                        basename = os.path.basename(os.path.join(output_dir, f))
                        tar.add(os.path.join(output_dir, f), arcname=basename)
                        os.remove(os.path.join(output_dir, f))
            logger.info(f"Done in {round(timer() - start)} sec")
            # add last downloaded pictures
            for f in os.listdir(output_dir):
                basename = os.path.basename(os.path.join(output_dir, f))
                tar.add(os.path.join(output_dir, f), arcname=basename)
                os.remove(os.path.join(output_dir, f))
            # remove empty directory
            if os.path.exists(output_dir):
                os.rmdir(output_dir)

