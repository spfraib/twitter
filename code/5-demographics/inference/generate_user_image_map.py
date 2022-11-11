import os
from pathlib import Path
import tarfile
import logging
import argparse
from tqdm import tqdm
import json
import pandas as pd


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str,
                        default="NG")
    args = parser.parse_args()
    return args

def generate_user_image_map(tar_dir):
    user_image_mapping_dict = dict()
    for tar_path in tqdm(list(Path(tar_dir).glob('*.tar'))):
        if 'err' not in tar_path.name:
            try:
                tfile = tarfile.open(tar_path, 'r', ignore_zeros=True)
                raw_paths_list = tfile.getnames()
                for raw_path in raw_paths_list:
                    filename = os.path.basename(raw_path)
                    filename_no_ext = os.path.splitext(filename)[0]
                    if filename_no_ext.isdigit():
                        uid = filename_no_ext
                    else:
                        logger.info(f'Filename {filename_no_ext} (original filename: {filename}) is not a user ID.')
                        uid = None
                    if uid:
                        user_image_mapping_dict[uid] = (tar_path.name, raw_path)
                    # saves <id>: (path_to_tar, file_member)
                    # Example: '1182331536': ('../resized_tars/BR/118.tar', '1182331536.jpeg'),
            except Exception as e:
                logger.info(f'Exception "{e}" when treating {tar_path.name}')
    return user_image_mapping_dict

def get_image_map(map_dict, user_id, index):
    if user_id in map_dict.keys():
        return map_dict[user_id][index]
    else:
        return None

if __name__ == '__main__':
    args = get_args_from_command_line()
    tar_dir = f'/scratch/spf248/twitter_data_collection/data/profile_pictures/{args.country_code}/tars'
    user_image_mapping_dict = generate_user_image_map(tar_dir=tar_dir)
    if args.country_code == 'NG':
        user_dir = f'/scratch/spf248/twitter_social_cohesion/data/preprocessed_from_twitter_api/profiles/NG'
    output_dir = f'/scratch/spf248/twitter_data_collection/data/user_timeline/profiles_with_tar_path/{args.country_code}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for path in Path(user_dir).glob('*.parquet'):
        df = pd.read_parquet(path)
        df = df[['user_id', 'user_name', 'user_screen_name', 'user_description', 'country_short', 'user_profile_image_url_https']]
        df['user_id'] = df['user_id'].astype(str)
        df['tfilename'] = df['user_id'].apply(lambda x: get_image_map(map_dict=user_image_mapping_dict, user_id=x, index=0))
        df['tmember'] = df['user_id'].apply(lambda x: get_image_map(map_dict=user_image_mapping_dict, user_id=x, index=1))
        output_filename = path.name.split("/")[-1]
        df.to_parquet(os.path.join(output_dir, output_filename))
    # output_dir = f'{tar_dir}/user_map_dict_all.json'
    # with open(output_dir, 'w') as fp:
    #     json.dump(user_image_mapping_dict, fp)
