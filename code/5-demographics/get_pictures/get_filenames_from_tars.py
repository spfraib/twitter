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
    parser.add_argument("--tar_type", type=str, help='Whether to look at the tar or resized_tars')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args_from_command_line()
    tar_folder_dict = {'normal': 'tars', 'resized': 'resized_tars'}
    data_path = f'/scratch/spf248/twitter/data/demographics/profile_pictures/{tar_folder_dict[args.tar_type]}/{args.country_code}'
    outfile_path = f'/scratch/spf248/twitter/data/demographics/profile_pictures/{tar_folder_dict[args.tar_type]}/list_files_{args.country_code}.txt.gz'
    outfile_err_path = f'/scratch/spf248/twitter/data/demographics/profile_pictures/{tar_folder_dict[args.tar_type]}/list_errors_{args.country_code}.txt.gz'
    if os.path.exists(outfile_path):
        os.remove(outfile_path)
    if os.path.exists(outfile_err_path):
        os.remove(outfile_err_path)
    for tar_path in Path(data_path).glob('*.tar'):
        logger.info(f'Saving file names from {tar_path}')
        if 'err' not in tar_path.name:
            tar_files = tarfile.open(tar_path)
            try:
                for member in tar_files.getmembers():
                    with gzip.open(outfile_path, 'at') as f:
                        print(member.name, file=f)
            except:
                logger.info(f'Error reading tar {tar_path}')
    # check errors
    total_err_user_id_list = list()
    input_err_path = os.path.join(data_path, 'err')
    for txt_file in Path(input_err_path).glob('*.txt'):
        try:
            f = open(txt_file, 'r')
            err_user_id_list = [err_str.split('\t')[0] for err_str in f.read().split('\n')]
            total_err_user_id_list += err_user_id_list
        except Exception as e:
            logger.info(f'Exception : {e}')
            logger.info(f'Error reading {txt_file.name}')
    total_err_user_id_list = list(set(total_err_user_id_list))
    with gzip.open(outfile_err_path, 'at') as f:
        for user_id in total_err_user_id_list:
            print(user_id, file=f)
