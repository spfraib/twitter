import gzip
import argparse
import logging
from pathlib import Path
import pandas as pd

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
    data_path = '/scratch/spf248/twitter/data'
    # get user id list
    user_data_path = f'{data_path}/user_timeline/user_timeline_crawled/{args.country_code}'
    user_df = pd.concat([pd.read_parquet(parquet_path) for parquet_path in Path(user_data_path).glob('*.parquet')])
    user_list = user_df['user_id'].unique().tolist()
    # get ids from users with downloaded pictures
    list_files_path = f'{data_path}/demographics/profile_pictures/tars/list_files_{args.country_code}.txt.gz'
    user_with_pictures_list = list()
    with gzip.open(list_files_path, 'rt') as f:
        for line in f:
            user_id_str = line.split('.')[0]
            if not user_id_str.isdigit():
                logger.info(f'{user_id_str} is not a digit')
            user_with_pictures_list.append(user_id_str)
    users_without_pictures_list = [user_id for user_id in user_list if user_id not in user_with_pictures_list]
    logger.info(f'# users without picture: {len(users_without_pictures_list)}')