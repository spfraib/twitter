# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import os
import tarfile
from glob import glob
import tempfile
import argparse

import numpy as np
from m3inference.preprocess import resize_imgs
import multiprocessing as mp


# %%
def get_env_var(varname,default):
    if os.environ.get(varname) != None:
        var = int(os.environ.get(varname))
        print(varname,':', var)
    else:
        var = default
        print(varname,':', var,'(Default)')
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


# %%
def get_hash(filename, hashsize = 3):
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
    


# %%
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



# %%
SLURM_JOB_ID            = get_env_var('SLURM_JOB_ID', 0)
SLURM_ARRAY_TASK_ID     = get_env_var('SLURM_ARRAY_TASK_ID', 0)
SLURM_ARRAY_TASK_COUNT  = get_env_var('SLURM_ARRAY_TASK_COUNT', 1)
# SLURM_JOB_CPUS_PER_NODE = get_env_var('SLURM_JOB_CPUS_PER_NODE', mp.cpu_count())

# %%
# Add arguments
parser = argparse.ArgumentParser("Tar and/or resize images to be used by the M3 package. Rembember to concatenate all files generated after running this script.")

parser.add_argument("--input_dir", type = str, 
                    default = "/scratch/spf248/twitter/data/profile_pictures/US/profile_pictures_sam/",
                    help = "where to original images are")

parser.add_argument("--output_dir", type = str, 
                    default = "/scratch/spf248/joao/data/profile_pictures/resized_tars/US/",
                    help = "where to store resized images for users in the target country")

parser.add_argument("--only_tar", type = bool, 
                    default = False,
                    help = "In case you do not want to resize the images, only Tar them")


# %%
args = parser.parse_args()
input_dir = args.input_dir
output_dir = args.output_dir
only_tar = args.only_tar

#args = parser.parse_args(args=[]) # in case you are running this in a notebook
#input_dir = "../serverdata/profile_pictures/BR/"
#output_dir = "../serverdata/profile_pictures/resized_tars/BR/"
#only_tar = False

# %%
cache = get_all_files_in_dir_with_tars(output_dir)
print("Cache has %d images." % (len(cache)))

# %%
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

all_dirs = get_all_dirs_recursively(input_dir)
selected_dirs = np.array_split(all_dirs, SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID]

for d in selected_dirs:
    if only_tar:
        transform_raw_to_tar(d, output_dir, SLURM_ARRAY_TASK_ID) 
    else:
        resize_and_tar_raw(d, output_dir, SLURM_ARRAY_TASK_ID, cache)    

# %% [markdown]
# Rembember to concatenate all files generated after running this script.

# %%
########

# for i in `seq 101 999`; do echo $i ; cat ${i}_*.tar > ${i}.tar; rm ${i}_*.tar;  done

#######
