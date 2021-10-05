import os
from pathlib import Path
import tarfile


def get_id_from_filename(raw_path):
    filename = os.path.basename(raw_path)
    filename_no_ext = os.path.splitext(filename)[0]
    if filename_no_ext.isdigit():
        return filename_no_ext
    else:

    return os.path.splitext(base)[0]

def generate_user_image_map(images_dir):
    user_image_mapping = dict()
    for tar_path in Path(images_dir).glob('*.tar'):
        if 'err' not in path.name:
            tfile = tarfile.open(tar_path, 'r', ignore_zeros=True)
            raw_paths_list = tfile.getnames()
            for raw_path in raw_paths_list:
                uid = get_id_from_filename(f)
                user_image_mapping[uid] = (tarf, f)
                # saves <id>: (path_to_tar, file_member)
                # Example: '1182331536': ('../resized_tars/BR/118.tar', '1182331536.jpeg'),

    return user_image_mapping