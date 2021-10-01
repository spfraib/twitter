import gzip
import argparse
import logging
from pathlib import Path
import pandas as pd
from collections import Counter
import os

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

if __name__ == '__main__':
    args = get_args_from_command_line()
    logger.info(f'Country code: {args.country_code}')
    data_path = '/scratch/spf248/twitter/data'
    # get user id list
    user_data_path = f'{data_path}/user_timeline/user_timeline_crawled/{args.country_code}'
    user_df = pd.concat([pd.read_parquet(parquet_path) for parquet_path in Path(user_data_path).glob('*.parquet')])
    user_list = user_df['user_id'].unique().tolist()
    logger.info(f'Total # of users: {len(user_list)}')
    # get ids from user for whom we got an error when trying to download their pictures
    list_errors_path = f'{data_path}/demographics/profile_pictures/tars/list_errors_{args.country_code}.txt.gz'
    user_id_errors_list = list()
    with gzip.open(list_errors_path, 'rt') as f:
        for line in f:
            line = line.replace('\n', '')
            if line.isdigit():
                user_id_errors_list.append(line)
            else:
                logger.info(f'User ID {line} from erroneous users is not a digit')
    logger.info(f'# of users whose picture we were not able to download: {len(user_id_errors_list)}')
    # get ids from users with downloaded pictures
    list_files_path = f'{data_path}/demographics/profile_pictures/tars/list_files_{args.country_code}.txt.gz'
    user_with_pictures_list = list()
    file_format_count_dict = dict()
    with gzip.open(list_files_path, 'rt') as f:
        for line in f:
            line = line.replace('\n', '')
            user_id_str = line.split('.')[0]
            file_format = os.path.splitext(line)[1].lower()
            if file_format not in file_format_count_dict.keys():
                file_format_count_dict[file_format] = 1
            else:
                file_format_count_dict[file_format] += 1
            if '(' in user_id_str:
                user_id_str = user_id_str.split('(')[0].replace(' ', '')
            if not user_id_str.isdigit():
                logger.info(f'{line} returns error')
            else:
                user_with_pictures_list.append(user_id_str)
    logger.info(f'File format repartition: {file_format_count_dict}')
    counter = Counter(user_with_pictures_list)
    logger.info(f'# unique user IDs with pictures: {len(list(counter))}')
    logger.info(f'# user IDs with more than one picture: {len([i for i in counter if counter[i]>1])}')
    pictures_and_error_list = user_with_pictures_list + user_id_errors_list
    users_without_pictures_list = [user_id for user_id in user_list if user_id not in pictures_and_error_list ]
    logger.info(f'# users without picture: {len(users_without_pictures_list)}')