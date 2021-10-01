import tarfile
import os
import argparse
from pathlib import Path
import gzip
import logging

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
    data_path = f'/scratch/spf248/twitter/data/demographics/data/profile_pictures/{tar_folder_dict[args.tar_type]}/{args.country_code}'
    outfile_path = f'/scratch/spf248/twitter/data/demographics/data/profile_pictures/tars/list_files_{args.country_code}.txt.gz'
    for tar_path in Path(data_path).glob('*.tar'):
        logger.info(f'Saving file names from {tar_path}')
        tar_files = tarfile.open(tar_path)
        for tar in tar_files.getmembers():
            with gzip.open(outfile_path, 'wt') as f:
                f.write(f'{tar.name}\n')