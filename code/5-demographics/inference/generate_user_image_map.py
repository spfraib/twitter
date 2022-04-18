import os
from pathlib import Path
import tarfile
import logging
import argparse
from tqdm import tqdm
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

if __name__ == '__main__':
    args = get_args_from_command_line()
    tar_dir = f'/scratch/spf248/twitter_data_collection/data/demographics/profile_pictures/tars'
    user_image_mapping_dict = generate_user_image_map(tar_dir=tar_dir)
    output_dir = f'{tar_dir}/user_map_dict_all.json'
    with open(output_dir, 'w') as fp:
        json.dump(user_image_mapping_dict, fp)
