import gzip
import argparse
import logging
from pathlib import Path
import pandas as pd
from collections import Counter
import os
import json

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
    # get user id list
    if args.country_code == 'NG':
        user_data_path = f'/scratch/spf248/twitter_social_cohesion/data/preprocessed_from_twitter_api/profiles/NG'
    elif args.country_code == 'US':
        user_data_path = f'/scratch/mt4493/twitter_labor/demographic_cls/US_profiles'
    # user_data_path = f'/scratch/spf248/twitter_data_collection/data/user_timeline/profiles'
    user_dict = pd.concat([pd.read_parquet(parquet_path, columns=['user_id', 'user_profile_image_url_https']) for parquet_path in Path(user_data_path).glob('*.parquet')]).set_index('user_id').to_dict()['user_profile_image_url_https']
    user_list = list(set(list(user_dict.keys())))
    # get ids from user for whom we got an error when trying to download their pictures
    list_errors_path = f'/scratch/mt4493/twitter_labor/demographic_cls/profile_pictures/{args.country_code}/tars/list_errors_all.txt.gz'
    user_id_errors_list = list()
    with gzip.open(list_errors_path, 'rt') as f:
        for line in f:
            user_id = line.replace('\n', '')
            if user_id.isdigit():
                user_id_errors_list.append(user_id)
            else:
                logger.info(f'User ID {user_id} from erroneous users is not a digit.')
    logger.info(f'Total # of users: {len(user_list)}')
    logger.info(f'# of users whose picture we were not able to download: {len(user_id_errors_list)}')
    # get ids from users with downloaded pictures
    list_files_path = f'/scratch/mt4493/twitter_labor/demographic_cls/profile_pictures/{args.country_code}/tars/list_files_all.txt.gz'
    user_with_pictures_list = list()
    file_format_count_dict = dict()
    with gzip.open(list_files_path, 'rt') as f:
        for line in f:
            line = line.replace('\n', '')
            filename = os.path.basename(line)
            user_id_str = filename.split('.')[0]
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
    logger.info(f'# user IDs with more than one picture: {len([i for i in counter if counter[i] > 1])}')
    pictures_and_error_list = user_with_pictures_list + user_id_errors_list
    users_without_pictures_list = list(set(user_list) - set(pictures_and_error_list))
    logger.info(f'# users without picture and not listed in errors: {len(users_without_pictures_list)}')
    if len(users_without_pictures_list) > 0:
        missing_pictures_dict = {k: user_dict[k] for k in users_without_pictures_list}
        missing_pictures_path = f'/scratch/mt4493/twitter_labor/demographic_cls/profile_pictures/{args.country_code}/tars/user_ids_w_missing_pics_all.json'
        if os.path.exists(missing_pictures_path):
            os.remove(missing_pictures_path)
        with open(missing_pictures_path, 'w') as fp:
            json.dump(missing_pictures_dict, fp)
