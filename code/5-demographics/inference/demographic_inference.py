import pandas as pd
import pyarrow.parquet as pq
import os
import json
from resize_images import resize_img
import argparse
from glob import glob
import multiprocessing as mp
from m3inference import M3Inference
import pprint
import numpy as np

languages = {'US': 'en', 'MX': 'es', 'BR': 'pt'}
user_data_dir = "/scratch/spf248/twitter/data/users/decahose/users_profile/"


def get_env_var(varname,default):
    
    if os.environ.get(varname) != None:
        var = int(os.environ.get(varname))
        print(varname,':', var)
    else:
        var = default
        print(varname,':', var,'(Default)')
    return var


# Choose Number of Nodes To Distribute Credentials: e.g. jobarray=0-4, cpu_per_task=20, credentials = 90 (<100)
SLURM_JOB_ID            = get_env_var('SLURM_JOB_ID',0)
SLURM_ARRAY_TASK_ID     = get_env_var('SLURM_ARRAY_TASK_ID',0)
SLURM_ARRAY_TASK_COUNT  = get_env_var('SLURM_ARRAY_TASK_COUNT',1)
SLURM_JOB_CPUS_PER_NODE = get_env_var('SLURM_JOB_CPUS_PER_NODE',mp.cpu_count())


# Add arguments
parser = argparse.ArgumentParser("Run demographic inference using M3 package")
parser.add_argument("--images_dir", type = str, help = "Directory containing images for users in the target country", default = "/scratch/spf248/twitter/data/classification/US/profile_pictures_sam/")
parser.add_argument("--resized_images_dir", type = str, help = "where to store resized images for users in the target country", default = "/scratch/spf248/twitter/data/classification/US/profile_pictures_resized_random/")
parser.add_argument("--output_dir", type = str, help = "where to save the inference results", default = "/scratch/spf248/twitter/data/classification/US/inference_results")
parser.add_argument("--country", help = "Name of the country: Enter US or MX or BR", default = "US")


# Parse Arguments
args = parser.parse_args()
images_dir = args.images_dir
resized_images_dir = args.resized_images_dir
country = args.country
output_dir = args.output_dir

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

result_file = open(os.path.join(output_dir,f"inference_{SLURM_ARRAY_TASK_ID}"),'w')

dirs = os.listdir(images_dir)

user_image_mapping = {}

for d in dirs:
    path = os.path.join(images_dir,d)
    files = os.listdir(path)
    for f in files:
        temp = f.split('.')
        if len(temp) == 2:
            user_image_mapping[temp[0]] = os.path.join(path,f)
    

def get_local_path(row):
    user = row['id']
    if user not in user_image_mapping:
        return "UNAVAILABLE"
    else:
        orig_path = user_image_mapping[user]
        image_name = orig_path.split("/")[-1]
        resized_path = os.path.join(resized_images_dir,image_name)
        resize_img(orig_path,resized_path)
        return resized_path



def set_lang(row):
	return languages[country]


def findMax(pred):
    label = None
    maxm = 0
    for key,val in pred.items():
        if val>maxm:
            maxm = val
            label = key
    return label
    
parquet_path = list(np.array_split(glob(os.path.join(user_data_dir,'*.parquet')),SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID])
df = pq.read_table(source=parquet_path).to_pandas()

df = df.rename(columns={'id_str': 'id', 'profile_image_url_https': 'img_path'})
df = df[['id','name','screen_name','description','lang','img_path']]
df['img_path'] = df.apply(get_local_path,axis=1)

df_filtered = df[df['img_path']!="UNAVAILABLE"]
df_filtered['lang']= df_filtered.apply(set_lang,axis=1)
df_json = json.loads(df_filtered.to_json(orient = 'records'))


# Run inference
m3 = M3Inference() # see docstring for details
predictions = m3.infer(df_json) # also see docstring for details

for item in predictions.items():
    user, pred = item
    gender = findMax(pred['gender'])
    age = findMax(pred['age'])
    org = findMax(pred['org'])
    res = f"{user}\t{gender}\t{age}\t{org}"
    result_file.write(res)

result_file.close()
