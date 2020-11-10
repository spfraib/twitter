# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import os
import sys
import socket
import re
import numpy as np
import string
import warnings
from timeit import default_timer as timer
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf,desc,row_number,col,year,month,dayofmonth,dayofweek,to_timestamp,size,isnan,when,count,col,count,lit,sum
import pyspark.sql.functions as F
from pyspark.sql.types import MapType, StringType, IntegerType, StructType, StructField, FloatType, ArrayType
from py4j.java_gateway import java_import
from functools import reduce
from pyspark.sql import DataFrame
# -

# Country: US
# Files: 37084
# Chunks: 10
# EXTRACT CHUNK 0
# TIME: 2606 SEC
# EXTRACT CHUNK 1
# ERROR WITH CHUNK 1
# EXTRACT CHUNK 2
# TIME: 1946 SEC
# EXTRACT CHUNK 3
# TIME: 2203 SEC
# EXTRACT CHUNK 4
# TIME: 2333 SEC
# EXTRACT CHUNK 5
# TIME: 3144 SEC
# EXTRACT CHUNK 6
# TIME: 3382 SEC
# EXTRACT CHUNK 7
# TIME: 2617 SEC
# EXTRACT CHUNK 8
# TIME: 1828 SEC
# EXTRACT CHUNK 9
# TIME: 2135 SEC

# # Config

country_code="FR"
print('Country:', country_code)

# +
try:
    spark
except NameError:
    if 'samuel' in socket.gethostname().lower():
        print('Create Local SparkSession')
        spark=SparkSession.builder.config("spark.driver.host", "localhost").appName("extract-timelines").getOrCreate()
    else:
        print('Create Cluster SparkSession')
        spark=SparkSession.builder.appName("extract-timelines").getOrCreate()
        
spark.conf.set("spark.sql.files.ignoreCorruptFiles", "true")
spark.conf.set('spark.sql.session.timeZone', 'UTC')
    
print('Hostname:', socket.gethostname())
if  'samuel' in socket.gethostname().lower():
    path_to_data='../../data'
else:
    path_to_data='/user/spf248/twitter/data'

# +
print('List files to be processed...')

fs=spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
list_status=fs.listStatus(spark._jvm.org.apache.hadoop.fs.Path(os.path.join(path_to_data,'timelines','historical','API',country_code)))
paths=[file.getPath().toString() for file in list_status]
paths=[path.replace('hdfs://dumbo','') for path in paths if 'json.bz2' in path]
np.random.seed(0)
paths=np.random.permutation(sorted(paths))

print('# Files:', len(paths))
# -

n_chunks=10
print('# Chunks:', n_chunks)
paths_chunks=np.array_split(paths, n_chunks)


# # Process Data

def extract_chunk(i_chunk,paths_chunk):

        df=spark.read.option(
        "compression","bzip2").option(
        "multiLine","true").option(
        "encoding","UTF-8").json(list(paths_chunk))
        
        df=df.select(
        'id_str',
        'created_at',
        'full_text',
        'lang',
        'user.id_str',
        'user.location',
        'coordinates.coordinates',
        'place.id',
        )

        df = df.toDF(*[
        'tweet_id',
        'created_at',
        'text',
        'tweet_lang',
        'user_id',
        'user_location',
        'tweet_coordinates',
        'place_id',
        ])

        df = df.withColumn('created_at', to_timestamp('created_at',"EEE MMM dd HH:mm:ss ZZZZZ yyyy"))
        df = df.withColumn('tweet_longitude', F.col('tweet_coordinates').getItem(0))
        df = df.withColumn('tweet_latitude',  F.col('tweet_coordinates').getItem(1))
        df = df.drop('tweet_coordinates')

        df.write.mode("overwrite").parquet(os.path.join(path_to_data,'timelines','historical','chunks',country_code,str(i_chunk)))


for i_chunk,paths_chunk in enumerate(paths_chunks):
    
    try:

        print('EXTRACT CHUNK', i_chunk)
        start = timer()

        extract_chunk(i_chunk,paths_chunk)

        end = timer()
        print('TIME:', round(end - start), 'SEC')

    except:

        print('ERROR WITH CHUNK', i_chunk)
        print('\n'.join(list(paths_chunk)))
