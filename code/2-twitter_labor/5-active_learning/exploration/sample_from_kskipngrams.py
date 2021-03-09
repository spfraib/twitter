# Change country_code + model_name
# sbatch --array=0-49 sample_from_kskipngrams.sh
import os
import sys
from time import time
from glob import glob
import numpy as np
import pandas as pd
import multiprocessing as mp
import argparse


# In[2]:


# Get Environment Variables
def get_env_var(varname,default):
    
    if os.environ.get(varname) != None:
        var = int(os.environ.get(varname))
        print(varname,':', var)
    else:
        var = default
        print(varname,':', var,'(Default)')
    return var

def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str)
    parser.add_argument("--model_folder", type=str)
    args = parser.parse_args()
    return args

SLURM_JOB_ID            = get_env_var('SLURM_JOB_ID',0)
SLURM_ARRAY_TASK_ID     = get_env_var('SLURM_ARRAY_TASK_ID',0)
SLURM_ARRAY_TASK_COUNT  = get_env_var('SLURM_ARRAY_TASK_COUNT',1)
SLURM_JOB_CPUS_PER_NODE = get_env_var('SLURM_JOB_CPUS_PER_NODE',mp.cpu_count())

args = get_args_from_command_line()
path_to_random_set = '/scratch/mt4493/twitter_labor/twitter-labor-data/data'
path_to_kskipngrams = '/scratch/mt4493/twitter_labor/twitter-labor-data/data/active_learning/k_skip_n_grams'
path_to_output = '/scratch/mt4493/twitter_labor/twitter-labor-data/data/active_learning/sampling_top_lift'
country_code = args.country_code
print('Country:', country_code)
model_name = args.model_folder
print('Model:', model_name)
n_sample = 5
print('# random tweets by ngram:', n_sample)


# In[4]:


kskipngrams = pd.read_json(os.path.join(path_to_kskipngrams,country_code,model_name,'kskipngrams_'+model_name+'.json'))
kskipngrams = kskipngrams.applymap(lambda x:tuple(x)).stack().drop_duplicates().tolist()
print('# ngrams:', len(kskipngrams))
if SLURM_ARRAY_TASK_ID>=len(kskipngrams):
    sys.exit('SLURM_ARRAY_TASK_ID out of bound')
tokens = kskipngrams[SLURM_ARRAY_TASK_ID]
print('tokens:', tokens)


# In[8]:


print('Load tweets...')
start_time = time()
tweets = pd.concat(pd.read_parquet(filename) for filename in glob(os.path.join(path_to_random_set,'random_samples','random_samples_splitted',country_code,'new_samples','*.parquet')))
tweets.set_index('tweet_id',inplace=True)
print('# tweets:', tweets.shape[0])
print('Time taken:', round(time() - start_time,1), 'seconds') # 82


# In[ ]:


def tokens2regex(tokens):
    regex = r''
    for i,token in enumerate(tokens):
        if not i:
            regex += r'\b' + token + r'\b'
        else:
            regex += r'.*\b' + token + r'\b'
    return regex


# In[ ]:


def sample_from_regex(regex):
    tmp = tweets[tweets['text'].str.contains(regex,case=False,regex='True')]
    tmp = tmp.sample(n = min(n_sample, tmp.shape[0]))
    tmp['regex'] = regex
    return tmp


# In[ ]:


print('Search pattern...')
start_time = time()
tweets_sampled = sample_from_regex(tokens2regex(tokens))
os.makedirs(os.path.join(path_to_output,country_code,model_name, 'subsamples'), exist_ok=True)
tweets_sampled.to_csv(os.path.join(path_to_output,country_code,model_name, 'subsamples', 'tweets_sampled_for_labeling_'+str(SLURM_JOB_ID)+'_'+str(SLURM_ARRAY_TASK_ID)+'.csv'))
print('Time taken:', round(time() - start_time,1), 'seconds')


print('Done !')


# # Finalize new sample to send to labeling

combined_sample = pd.concat([pd.read_csv(filename) for filename in glob(os.path.join(path_to_output, country_code,model_name,'subsamples','tweets_sampled_for_labeling_*.csv'))]).drop_duplicates('tweet_id').drop_duplicates('text').set_index('tweet_id')
combined_sample.to_csv(os.path.join(path_to_output, country_code,model_name,'tweets_sampled_for_labeling_'+model_name+'.csv'))
print('# Explored tweets:', combined_sample.shape[0])


# In[5]:


combined_sample.groupby('regex')['text'].count()

