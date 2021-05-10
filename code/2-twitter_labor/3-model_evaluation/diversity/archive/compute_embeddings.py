import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
from glob import glob
import getpass


def get_env_var(varname, default):
    if os.environ.get(varname) != None:
        var = int(os.environ.get(varname))
        print(varname, ':', var)
    else:
        var = default
        print(varname, ':', var, '(Default)')
    return var


# to load

# data2=np.load('/scratch/mt4493/test/embeddings_test.npy', allow_pickle=True)
# data2[()] -> dict

if __name__ == '__main__':
    # define vars
    SLURM_ARRAY_TASK_ID = get_env_var('SLURM_ARRAY_TASK_ID', 0)
    SLURM_ARRAY_TASK_COUNT = get_env_var('SLURM_ARRAY_TASK_COUNT', 1)
    SLURM_JOB_ID = get_env_var('SLURM_JOB_ID', 1)
    path_to_data = '/scratch/mt4493/twitter_labor/twitter-labor-data/data/random_samples/random_samples_splitted/US'
    paths_to_random = list(np.array_split(
        glob(os.path.join(path_to_data, 'evaluation', '*.parquet')),
        SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID])
    # load tweets
    random_df = pd.concat([pd.read_parquet(file)[['tweet_id', 'text']] for file in paths_to_random])
    tweet_list = random_df['text'].tolist()
    tweet_id_list = random_df['tweet_id'].tolist()
    # load encoder, encode and store in dict (key: tweet_id, value: embedding)
    diversity_model = SentenceTransformer('stsb-roberta-large', device='cuda')
    embeddings = diversity_model.encode(tweet_list, convert_to_numpy=True)
    embeddings_dict = dict(zip(tweet_id_list, embeddings))
    # save dict
    output_folder_path = os.path.join(path_to_data, 'evaluation_embeddings')
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    output_path = os.path.join(output_folder_path, f'{getpass.getuser()}_random-{str(SLURM_ARRAY_TASK_ID)}.npy')
    np.save(output_path, embeddings_dict)
