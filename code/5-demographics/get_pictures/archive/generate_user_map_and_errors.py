import tarfile
import os
import argparse
from pathlib import Path
import gzip
import logging
from PIL import Image
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
    parser.add_argument("--tar_type", type=str, help='Whether to look at the tar or resized_tars')
    args = parser.parse_args()
    return args


def generate_user_image_map(tar_dir, outfile_path):
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
                        uid_dict = dict()
                        uid_dict[uid] = (tar_path.name, raw_path)
                        json_str = f'{json.dumps(uid_dict)}\n'
                        json_bytes = json_str.encode('utf-8')
                        with gzip.open(outfile_path, 'at') as f:
                            f.write(json_bytes)
                    # saves <id>: (path_to_tar, file_member)
                    # Example: '1182331536': ('../resized_tars/BR/118.tar', '1182331536.jpeg'),
            except Exception as e:
                logger.info(f'Exception "{e}" when treating {tar_path.name}')


def generate_error_list(tar_dir, outfile_err_path):
    total_err_user_id_list = list()
    for tar_path in tqdm(list(Path(tar_dir).glob('err*.tar'))):
        tar_files = tarfile.open(tar_path)
        for member in tar_files.getmembers():
            try:
                f = tar_files.extractfile(member)
                err_user_id_list = [err_str.split('\t')[0] for err_str in f.read().decode('utf-8').split('\n')]
                total_err_user_id_list += err_user_id_list
            except:
                logger.info(f'Error reading member {member.name} from {tar_path.name}')
    with gzip.open(outfile_err_path, 'at') as f:
        for user_id in total_err_user_id_list:
            print(user_id, file=f)


if __name__ == '__main__':
    args = get_args_from_command_line()
    tar_folder_dict = {'normal': 'tars', 'resized': 'resized_tars'}
    data_path = f'/scratch/spf248/twitter/data/demographics/profile_pictures/{tar_folder_dict[args.tar_type]}/{args.country_code}'
    outfile_path = f'/scratch/spf248/twitter/data/demographics/profile_pictures/{tar_folder_dict[args.tar_type]}/user_image_map_{args.country_code}.json.gz'
    outfile_err_path = f'/scratch/spf248/twitter/data/demographics/profile_pictures/{tar_folder_dict[args.tar_type]}/list_errors_{args.country_code}.txt.gz'
    if os.path.exists(outfile_path):
        os.remove(outfile_path)
    if os.path.exists(outfile_err_path):
        os.remove(outfile_err_path)
    logger.info('Starting to generate user image map')
    generate_user_image_map(tar_dir=data_path, outfile_path=outfile_path)
    logger.info('Starting to generate error list')
    generate_error_list(tar_dir=data_path, outfile_err_path=outfile_err_path)
