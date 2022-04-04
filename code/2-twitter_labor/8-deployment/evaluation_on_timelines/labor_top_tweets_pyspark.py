#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
from datetime import datetime
import re
import numpy as np
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.types import *


# In[ ]:


print('Date:',datetime.today().strftime('%Y-%m-%d'))
spark = SparkSession.builder.config('spark.driver.extraJavaOptions', '-Duser.timezone=UTC') .config('spark.executor.extraJavaOptions', '-Duser.timezone=UTC') .config('spark.sql.session.timeZone', 'UTC') .getOrCreate()
print(spark.sparkContext.getConf().getAll())
path_to_data = '/user/spf248/twitter/data'
print(path_to_data)
cutoff_scores = spark.read.option("header", "true").csv(os.path.join(path_to_data,'labor','labor_BERT_cutoff_scores.csv'))
cutoff_scores = cutoff_scores.toPandas()
country_codes = ["US","BR","MX"]
classes = ['is_hired_1mo','is_unemployed','job_offer','job_search','lost_job_1mo']
print('Classes:', classes)


# In[ ]:


def listdir(path2dir):
    hadoop = spark.sparkContext._jvm.org.apache.hadoop
    fs = hadoop.fs.FileSystem
    conf = hadoop.conf.Configuration()
    paths = fs.get(conf).globStatus(hadoop.fs.Path(path2dir))
    paths = [ f.getPath().toString() for f in paths ]
    return paths

def get_BERT_iter(country_code):
    return str(np.max([int(re.findall('iter_(\d+)-',x)[0]) for x in listdir(os.path.join(path_to_data,'labor','tweet_scores','BERT',country_code,'*'))]))
    
def get_path_to_scores(country_code, model):
    if model == 'BERT':
        return os.path.join(path_to_data,'labor','tweet_scores',model,country_code,'iter_'+get_BERT_iter(country_code)+'*','output')
    elif model == 'keywords':
        return os.path.join(path_to_data,'labor','tweet_scores',model,country_code)
    
def get_class2cutoff(country_code, model, cutoff_proba = 1):
    if model == 'keywords':
        return dict(zip(classes,[1]*len(classes)))
    elif model == 'BERT':
        return cutoff_scores.loc[(cutoff_scores['country']==country_code)&(cutoff_scores['cutoff_proba']==cutoff_proba)].set_index('class')['iter'+get_BERT_iter(country_code)].astype(float).to_dict()

def get_model_name(country_code, model, cutoff_proba = 1):
    if model == 'keywords':
        return model
    elif model == 'BERT':
        return model+'_iter'+get_BERT_iter(country_code)+'_cutoff_proba_'+cutoff_proba
    
def get_top_tweets(country_code, class_, model = 'BERT', cutoff_proba = '0.5'):
    class2cutoff = get_class2cutoff( country_code, model, cutoff_proba )
    model_name = get_model_name(country_code, model, cutoff_proba)
    user_timeline_extracts = spark.read.parquet( os.path.join( path_to_data, 'user_timeline', 'user_timeline_extracts', country_code ) )
    tweet_scores = spark.read.parquet( get_path_to_scores( country_code, model ) )
    tmp = tweet_scores.filter( tweet_scores[class_] >= class2cutoff[class_] ).select('tweet_id', class_).join( user_timeline_extracts, on = 'tweet_id' )
    tmp = tmp.drop('user_mentions', 'user_retweeted')
    tmp.write.mode("overwrite").parquet( os.path.join( path_to_data, 'labor', 'top_tweets', country_code, model_name, class_ ) )


# In[ ]:


for country_code in country_codes:
    print(country_code)
    for class_ in classes:
        get_top_tweets(country_code, class_)

