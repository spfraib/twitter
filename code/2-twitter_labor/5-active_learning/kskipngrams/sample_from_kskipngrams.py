#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
from time import time
from glob import glob
import numpy as np
import pandas as pd
import multiprocessing as mp


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

SLURM_JOB_ID            = get_env_var('SLURM_JOB_ID',0)
SLURM_ARRAY_TASK_ID     = get_env_var('SLURM_ARRAY_TASK_ID',0)
SLURM_ARRAY_TASK_COUNT  = get_env_var('SLURM_ARRAY_TASK_COUNT',1)
SLURM_JOB_CPUS_PER_NODE = get_env_var('SLURM_JOB_CPUS_PER_NODE',mp.cpu_count())


# In[23]:


path_to_input = '/scratch/mt4493/twitter_labor/twitter-labor-data/data'
path_to_output = '/scratch/spf248/twitter/data'
country_code = 'US'
print('Country:', country_code)
model_name = 'iter_0-convbert-1122153'
print('Model:', model_name)
labels_name = 'jan5_iter0'
print('Previous labels:', labels_name)
n_sample = 10
print('# random tweets by ngram:', n_sample)


# In[4]:


kskipngrams = pd.read_pickle(os.path.join(path_to_output,'active_learning',country_code,model_name,'kskipngrams.pkl'))
kskipngrams = kskipngrams.stack().drop_duplicates().tolist()
print('# ngrams:', len(kskipngrams))
if SLURM_ARRAY_TASK_ID>=len(kskipngrams):
    sys.exit('SLURM_ARRAY_TASK_ID out of bound')
tokens = kskipngrams[SLURM_ARRAY_TASK_ID]
print('tokens:', tokens)


# In[5]:


print('Load tweets...')
start_time = time()
with mp.Pool() as pool:
    tweets = pd.concat(pool.map(pd.read_parquet, glob(os.path.join(path_to_input,'random_samples','random_samples_splitted',country_code,'new_samples','*.parquet'))))
tweets.set_index('tweet_id',inplace=True)
print('# tweets:', tweets.shape[0])
print('Time taken:', round(time() - start_time,1), 'seconds') # 82


# In[6]:


def tokens2regex(tokens):
    regex = r''
    for i,token in enumerate(tokens):
        if not i:
            regex += r'\b' + token + r'\b'
        else:
            regex += r'.*\b' + token + r'\b'
    return regex


# In[7]:


def sample_from_regex(regex):
    tmp = tweets[tweets['text'].str.contains(regex,case=False,regex='True')]
    tmp = tmp.sample(n = min(n_sample, tmp.shape[0]))
    tmp['regex'] = regex
    return tmp


# In[8]:


print('Search pattern...')
start_time = time()
tweets_sampled = sample_from_regex(tokens2regex(tokens))
os.makedirs(os.path.join(path_to_output,'active_learning',country_code,model_name), exist_ok=True)
tweets_sampled.to_csv(os.path.join(path_to_output,'active_learning',country_code,model_name,'tweets_sampled_for_labeling_'+str(SLURM_JOB_ID)+'_'+str(SLURM_ARRAY_TASK_ID)+'.csv'))
print('Time taken:', round(time() - start_time,1), 'seconds')


# In[ ]:


print('Done !')


# # Finalize new sample to send to labeling

# In[57]:


combined_sample = pd.concat([pd.read_csv(filename) for filename in glob(os.path.join(path_to_output,'active_learning',country_code,model_name,'tweets_sampled_for_labeling_*.csv'))]).drop_duplicates('tweet_id').drop_duplicates('text').set_index('tweet_id')
combined_sample.to_csv(os.path.join(path_to_output,'active_learning',country_code,model_name,'tweets_sampled_for_labeling.csv'))


# In[58]:


combined_sample.groupby('regex')['text'].count().sort_values()


# In[62]:


combined_sample.groupby('regex').head(1).head(50)


# In[ ]:




