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
    for tar_path in Path(data_path).glob('*.tar'):
        if tar_path.name != 'err.tar':
            logger.info(f'Saving file names from {tar_path}')
            tar_files = tarfile.open(tar_path)
            for member in tar_files.getmembers():
                with gzip.open(outfile_path, 'at') as f:
                    print(member.name, file=f)
        else:
            tar_files = tarfile.open(tar_path)
            total_err_user_id_list = list()
            for member in tar_files.getmembers():
                f = tar_files.extractfile(member)
                err_user_id_list = [err_str.split('\t')[0] for err_str in f.read().decode('utf-8').split('\n')]
                total_err_user_id_list += err_user_id_list
            with gzip.open(outfile_err_path, 'at') as f:
                for user_id in total_err_user_id_list:
                    print(user_id, file=f)
