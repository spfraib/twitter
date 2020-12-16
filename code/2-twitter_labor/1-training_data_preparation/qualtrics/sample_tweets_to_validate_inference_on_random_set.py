#!/usr/bin/env python
# coding: utf-8

# In[16]:


import os
import numpy as np
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql import Window
import argparse


spark = SparkSession.builder.appName("").getOrCreate()

def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str)
    args = parser.parse_args()
    return args

args = get_args_from_command_line()
country_code='US'
label=args.label
path_to_tweets='/user/mt4493/twitter/random_samples/random_samples_splitted/US/evaluation' # Random set of tweets
path_to_scores= os.path.join('/user/mt4493/twitter/inference/US/DeepPavlov_bert-base-cased-conversational_nov13_iter0_14045091-14114233-evaluation/output', label) # Prediction scores from classification
path_to_evals='/user/mt4493/twitter/evaluation/US'  # Where to store the sampled tweets to be labeled


# In[18]:


def get_sampled_indices(n_sample=10,n_cutoff=6):
    sampled_points=[] # index of scores around which we sample n_sample tweets
    sampled_ranks=[] # ranks of sampled tweets
    for point,rank in enumerate(sorted(set([int(x) for i in range(n_cutoff) for x in np.logspace(i,i+1,i+1)]))):
        if not point:
            new_ranks=list(range(rank,rank+n_sample))
        else:
            new_ranks=list(range(rank+1,rank+n_sample+1))
        print('Index of sampled point:', point)
        print('Sampled ranks:', new_ranks)
        sampled_points.extend([point]*n_sample)
        sampled_ranks.extend(new_ranks)
    return sampled_points, sampled_ranks

sampled_points, sampled_ranks=get_sampled_indices()
print('# Sampled points:', len(set(sampled_points)))
print('# Sampled tweets:', len(sampled_ranks))


# In[ ]:


tweets=spark.read.parquet(os.path.join(path_to_tweets))
scores=spark.read.parquet(os.path.join(path_to_scores))
sampled_indices=spark.createDataFrame(zip(sampled_points, sampled_ranks), schema=['point', 'rank'])


# In[ ]:


df=tweets.select('tweet_id','text').join(scores.select('tweet_id','score'),on='tweet_id')
df=df.withColumn("rank", F.row_number().over(Window.orderBy(F.desc("score"))))
df=df.join(sampled_indices,on='rank')


# In[ ]:


df.coalesce(1).write.option("header", "true").csv(os.path.join(path_to_evals))

