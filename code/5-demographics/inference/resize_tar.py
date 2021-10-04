import os
import tarfile
from glob import glob
import tempfile
import argparse

import numpy as np
from m3inference.preprocess import resize_imgs
import multiprocessing as mp


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str)
    parser.add_argument("--output_dir", type=str,
                        default="/scratch/spf248/joao/data/profile_pictures/resized_tars/US/",
                        help="where to store resized images for users in the target country")

    parser.add_argument("--resize_before_tar", type=bool,
                        default=True,
                        help="This option will both resie and tar the images, otherwise, it will only tar them.")
    args = parser.parse_args()
    return args


def get_env_var(varname, default):
    if os.environ.get(varname) != None:
        var = int(os.environ.get(varname))
        print(varname, ':', var)
    else:
        var = default
        print(varname, ':', var, '(Default)')
    return var


def reset(tarinfo):
    tarinfo.uid = tarinfo.gid = 0
    tarinfo.uname = tarinfo.gname = "root"
    return tarinfo


def files_to_tarfile(tfile, filelist, mode="a"):
    if mode == "w":
        print('Creating new tar file `%s`' % (tfile))
    elif mode == "a":
        print('Trying to append %d files to tar file `%s`' % (len(filelist), tfile))

    with tarfile.open(tfile, mode=mode, ignore_zeros=True) as tar:
        for f in filelist:
            if not check_if_file_in_tarfile(tar, os.path.basename(f)):
                tar.add(f, filter=reset, arcname=os.path.basename(f))


def check_if_file_in_tarfile(tfile, filename):
    if isinstance(tfile, str):
        tfile = tarfile.open(tfile, 'r', ignore_zeros=True)
    return filename in tfile.getnames()


def get_all_dirs_recursively(directory):
    return [x[0] for x in os.walk(directory)]

def get_hash(filename, hashsize=3):
    """
    Current hash is just getting the first 3 letters in the file name.
    Change it to something else if needed.
    """
    name = os.path.basename(filename)
    return name[:hashsize]


def create_dict_mapping_from_folder(folder):
    d = {}
    files = glob(os.path.join(folder, "*"))
    for f in files:
        if not os.path.isfile(f):
            continue

        h = get_hash(f)
        if h not in d:
            d[h] = []
        d[h].append(f)
    return d


def transform_raw_to_tar(input_dir, output_dir, task_id):
    hmap = create_dict_mapping_from_folder(input_dir)
    for k, filelist in hmap.items():
        tfile = os.path.join(output_dir, "%s_%s.tar" % (k, task_id))
        if not os.path.exists(tfile):
            files_to_tarfile(tfile, filelist, "w")
        else:
            files_to_tarfile(tfile, filelist, "a")

    print("Done! Tar files were created.")


def get_all_files_in_tar(tarf):
    with tarfile.open(tarf, 'r', ignore_zeros=True) as tfile:
        return tfile.getnames()


def get_all_files_in_dir_with_tars(directory):
    known_ids = set([])
    for d in get_all_dirs_recursively(directory):
        for tfile in glob(os.path.join(d, "*.tar")):
            known_ids.update(set(get_all_files_in_tar(tfile)))
    return known_ids

def resize_and_tar_raw(input_dir, output_dir, task_id, cache):
    with tempfile.TemporaryDirectory() as temp:
        non_processed_files = []
        all_files = glob(os.path.join(input_dir, "*"))
        for file in all_files:
            if os.path.basename(file) not in cache:
                non_processed_files.append(file)

        if len(non_processed_files) < len(all_files):
            print("Dir %s was already processed before" % (input_dir))
            return

        resize_imgs(input_dir, temp)
        transform_raw_to_tar(temp, output_dir, task_id)


if __name__ == '__main__':
    args = get_args_from_command_line()

    SLURM_JOB_ID = get_env_var('SLURM_JOB_ID', 0)
    SLURM_ARRAY_TASK_ID = get_env_var('SLURM_ARRAY_TASK_ID', 0)
    SLURM_ARRAY_TASK_COUNT = get_env_var('SLURM_ARRAY_TASK_COUNT', 1)

    input_dir = f'/scratch/spf248/twitter/data/demographics/profile_pictures/tars/{args.country_code}'

    cache = get_all_files_in_dir_with_tars(args.output_dir)
    print("Cache has %d images." % (len(cache)))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    all_dirs = get_all_dirs_recursively(args.input_dir)
    selected_dirs = np.array_split(all_dirs, SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID]

    print("Selected dirs:", selected_dirs)

    for d in selected_dirs:
        if args.resize_before_tar:
            resize_and_tar_raw(d, args.output_dir, SLURM_ARRAY_TASK_ID, cache)
        else:
            transform_raw_to_tar(d, args.output_dir, SLURM_ARRAY_TASK_ID)

# %% [markdown]
# Remember to concatenate all files generated after running this script.

# %%
########

# for i in `seq 100 999`; do echo $i ; cat ${i}_*.tar > ${i}.tar; rm ${i}_*.tar;  done
# for i in `seq 100 999`; do mkdir d${i};  tar -C d${i} -ixf ${i}.tar; cd d${i};  tar -zcf ../d${i}.tar *; cd -; rm -rf d${i}; echo "Done ${i}"; done

# Both commands together:
# for i in `seq 100 999`; do mkdir d${i}; cat ${i}_*.tar > ${i}.tar; tar -C d${i} -ixf ${i}.tar; cd d${i}; tar -zcf ../${i}.tar *; cd -; rm -rf d${i}; echo "Done ${i}"; done

#######
