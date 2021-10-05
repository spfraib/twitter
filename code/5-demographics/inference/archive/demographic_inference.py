import pprint
import uuid
import tarfile
import tempfile
import os
import json
import argparse

import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import multiprocessing as mp

from glob import glob
from m3inference import M3Inference

# -

def get_env_var(varname,default):
    if os.environ.get(varname) != None:
        var = int(os.environ.get(varname))
        print(varname,':', var)
    else:
        var = default
        print(varname,':', var,'(Default)')
    return var


# Choose Number of Nodes To Distribute Credentials: e.g. jobarray=0-4, cpu_per_task=20, credentials = 90 (<100)
SLURM_JOB_ID            = get_env_var('SLURM_JOB_ID', 0)
SLURM_ARRAY_TASK_ID     = get_env_var('SLURM_ARRAY_TASK_ID', 0)
SLURM_ARRAY_TASK_COUNT  = get_env_var('SLURM_ARRAY_TASK_COUNT', 1)
SLURM_JOB_CPUS_PER_NODE = get_env_var('SLURM_JOB_CPUS_PER_NODE', mp.cpu_count())


# +
# Add arguments
languages = {'US': 'en', 'MX': 'es', 'BR': 'pt'}

parser = argparse.ArgumentParser("Run demographic inference using M3 package")

parser.add_argument("--user_data_dir", type = str, 
                    default = "/scratch/spf248/twitter/data/user_timeline/user_timeline_crawled/US/",
                    help = "where to store resized images for users in the target country")

parser.add_argument("--resized_images_dir", type = str, 
                    default = "/scratch/spf248/joao/data/profile_pictures/resized_tars/US/",
                    help = "where to store resized images for users in the target country")

parser.add_argument("--output_dir", type = str, 
                    default = "/scratch/spf248/joao/data/inference_results/US/",
                    help = "where to save the inference results")

parser.add_argument("--country", help = "Name of the country: Enter US or MX or BR", default = "US")


# +
# Parse Arguments
args = parser.parse_args()
resized_images_dir = args.resized_images_dir
country = args.country
output_dir = args.output_dir
user_data_dir = args.user_data_dir

#args = parser.parse_args(args=[])
#resized_images_dir = "../serverdata/profile_pictures/resized_tars/BR/"
#country = "BR"
#output_dir = "../serverdata/inference_results/BR/"
#user_data_dir = "../serverdata/user_timeline_crawled/BR/"


# +
def set_lang(row):
    languages = {'US': 'en', 'MX': 'es', 'BR': 'pt'}
    return languages[country]

def findMax(pred):
    label = None
    maxm = 0
    for key,val in pred.items():
        if val>maxm:
            maxm = val
            label = key
    return label


# +
def get_files_from_tar(tfile):
    if isinstance(tfile, str):
        tfile = tarfile.open(tfile, 'r', ignore_zeros=True)
    return tfile.getnames()

def get_id_from_filename(filename):
    base = os.path.basename(filename)
    return os.path.splitext(base)[0]


# -

def generate_user_image_map(images_dir):
    user_image_mapping = {}
    for d in os.listdir(images_dir):
        tarf = os.path.join(images_dir, d)
        files = get_files_from_tar(tarf)
        
        for f in files:
            uid = get_id_from_filename(f)
            user_image_mapping[uid] = (tarf, f)
            # saves <id>: (path_to_tar, file_member)
            # Example: '1182331536': ('../resized_tars/BR/118.tar', '1182331536.jpeg'),
            
    return user_image_mapping



def get_local_path(row, tmpdir, user_image_mapping):
    user = row['id']
    if user not in user_image_mapping:
        return np.nan
    else:
        tmpd = tempfile.TemporaryDirectory()
        tfilename, tmember = user_image_mapping[user]
        with tarfile.open(tfilename, mode='r', ignore_zeros=True) as tarf:
            tarf.extract(tmember, path=tmpdir)

            
        return os.path.join(tmpdir, tmember)


# +
# Load previous inferences, if they exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
files_on_output = glob(os.path.join(output_dir, "*"))

if len(files_on_output) > 0:
    previous_result = pd.concat([pd.read_csv(f) for f in files_on_output if f.endswith(".csv.gz")])
    known_ids = set(map(str, previous_result["user"].unique()))
    print("A total of %d ids were already known." % (len(known_ids))) 
else:
    known_ids = set([])
# -

user_image_mapping = generate_user_image_map(resized_images_dir)

# +
selected_parquets = list(np.array_split(glob(os.path.join(user_data_dir,'*.parquet')), SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID])

if len(selected_parquets) == 0:
    print("ERROR: NO PARQUETS WERE FOUND!")

print("Reading %d parquets" % (len(selected_parquets)))

df = pq.read_table(source=selected_parquets).to_pandas()

# Remove uids that we had already calculated before
df = df[~df["user_id"].isin(known_ids)]
print("Going to process the remaining %d user ids" % (df.shape[0]))

# +
df = df.rename(columns={'user_id': 'id', 'profile_image_url_https': 'img_path'})
df = df[['id', 'name', 'screen_name', 'description', 'lang', 'img_path']]
df['lang']= df.apply(set_lang,axis=1)

for (ichunk, chunk) in enumerate(np.array_split(df, 100)):
    
    if chunk.shape[0] == 0:
        continue
    
    with tempfile.TemporaryDirectory() as tmpd:
        chunk['img_path'] = chunk.apply(lambda x: get_local_path(x, tmpd, user_image_mapping), axis=1)
        chunk = chunk.dropna()

        if chunk.shape[0] == 0:
            continue

        df_json = json.loads(chunk.to_json(orient = 'records'))
        print("Procesing chunk %d -- another %d users" % (ichunk, len(df_json)))

        # Run inference
        m3 = M3Inference() # see docstring for details
        predictions = m3.infer(df_json) # also see docstring for details
# -

        if predictions:
            result_file = os.path.join(output_dir, "processed_%s.csv.gz" % str(uuid.uuid4()))

            rows = []
            for item in predictions.items():
                user, pred = item
                row = {"user": user, "gender": findMax(pred['gender']), 
                       "age": findMax(pred['age']), "org": findMax(pred['org'])}
                rows.append(row)   

            pd.DataFrame(rows).to_csv(result_file, index=False)
print("Process completed!")
