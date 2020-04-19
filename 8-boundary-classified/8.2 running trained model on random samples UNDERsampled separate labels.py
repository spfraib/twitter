#!/usr/bin/env python
# coding: utf-8

# run using: sbatch --array=0-9 7.9-get-predictions-from-BERT.sh



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


# In[ ]:


path_to_data='/scratch/spf248/twitter/data/classification/US/'


# In[ ]:


print('Load Filtered Tweets:')
# filtered contains 8G of data!!
start_time = time.time()

paths_to_filtered=list(np.array_split(
glob(os.path.join(path_to_data,'filtered','*.parquet')),SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID])
print('#files:', len(paths_to_filtered))

tweets_filtered=pd.DataFrame()
for file in paths_to_filtered:
    tweets_filtered=pd.concat([tweets_filtered,pd.read_parquet(file)[['tweet_id','text']]])

print('time taken to load keyword filtered sample:', str(time.time() - start_time), 'seconds')
print(tweets_filtered.shape)


# In[ ]:


print('Load Random Tweets:')
# random contains 7.3G of data!!
start_time = time.time()

paths_to_random=list(np.array_split(
glob(os.path.join(path_to_data,'random','*.parquet')),SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID])
print('#files:', len(paths_to_random))

tweets_random=pd.DataFrame()
for file in paths_to_random:
    tweets_random=pd.concat([tweets_random,pd.read_parquet(file)[['tweet_id','text']]])

print('time taken to load random sample:', str(time.time() - start_time), 'seconds')
print(tweets_random.shape)


# In[ ]:


print('Predictions of Filtered Tweets:')
start_time = time.time()
predictions_filtered = learner.predict_batch(tweets_filtered['text'].values.tolist())
print('time taken:', str(time.time() - start_time), 'seconds')


# In[ ]:


print('Predictions of Random Tweets:')
start_time = time.time()
predictions_random = learner.predict_batch(tweets_random['text'].values.tolist())
print('time taken:', str(time.time() - start_time), 'seconds')


# In[ ]:


print('Save Predictions of Filtered Tweets:')
start_time = time.time()

df_filtered = pd.DataFrame(
[dict(prediction) for prediction in predictions_filtered],
index=tweets_filtered.tweet_id).rename(columns={
'is_unemployed':'unemployed',
'job_search':'search',
'is_hired_1mo':'hired',
'lost_job_1mo':'loss',
'job_offer"':'offer',
})

df_filtered.to_csv(
os.path.join(root_path,'pred','filtered'+'-'+str(SLURM_JOB_ID)+'-'+str(SLURM_ARRAY_TASK_ID)+'.csv'))

print('time taken:', str(time.time() - start_time), 'seconds')


# In[ ]:


print('Save Predictions of Random Tweets:')
start_time = time.time()

df_random = pd.DataFrame(
[dict(prediction) for prediction in predictions_random],
index=tweets_random.tweet_id).rename(columns={
'is_unemployed':'unemployed',
'job_search':'search',
'is_hired_1mo':'hired',
'lost_job_1mo':'loss',
'job_offer"':'offer',
})

df_random.to_csv(
os.path.join(root_path,'pred','random'+'-'+str(SLURM_JOB_ID)+'-'+str(SLURM_ARRAY_TASK_ID)+'.csv'))

print('time taken:', str(time.time() - start_time), 'seconds')

