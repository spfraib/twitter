import tarfile
import os
import argparse
from pathlib import Path
import gzip
import logging
from PIL import Image

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str,
                        default="US")
    parser.add_argument("--tar_type", type=str, help='Whether to look at the tar or resized_tars', default='normal')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args_from_command_line()
    logger.info(f'Country code: {args.country_code}')
    tar_folder_dict = {'normal': 'tars', 'resized': 'resized_tars'}
    data_path = f'/scratch/spf248/twitter/data/demographics/profile_pictures/{tar_folder_dict[args.tar_type]}/{args.country_code}'
    to_delete_dict = dict()
    for tar_path in Path(data_path).glob('*.tar'):
        if tar_path.name != 'err.tar':
            tar_files = tarfile.open(tar_path)
            for member in tar_files.getmembers():
                to_delete_dict[tar_path.name] = list()
                if '(' in member.name:
                    to_delete_dict[tar_path.name].append(member.name)
                f = tar_files.extractfile(member)
                content = Image.open(f)
                if content.size[0] < 224:
                    if not member.name in to_delete_dict[tar_path.name]:
                        to_delete_dict[tar_path.name].append(member.name)
    count = 0
    for tar_path_name in to_delete_dict.keys():
        for member_name in to_delete_dict[tar_path_name]:
            logger.info(f'Deleting {member_name} in {tar_path_name}')
            os.system(f'tar -vf {os.path.join(data_path, tar_path_name)} --delete {member_name}')
            count += 1
    logger.info(f'Deleted {count} images')
