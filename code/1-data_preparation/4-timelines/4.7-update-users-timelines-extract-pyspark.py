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
from glob import glob

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf,desc,row_number,col,year,month,dayofmonth,dayofweek,to_timestamp,size,isnan,when,count,col,count,lit,sum
import pyspark.sql.functions as F
from pyspark.sql.types import MapType,StringType,IntegerType,StructType,StructField,FloatType,ArrayType
from py4j.java_gateway import java_import
from functools import reduce
from pyspark.sql import DataFrame

# +
country_code = "MX"
print('Country:', country_code)

this_batch = '062020'
print('This batch:', this_batch)

# +
try:
    spark
except NameError:
    print('Create Spark')
    spark=SparkSession.builder.appName("").getOrCreate()
    
# IgnoreCorruptFiles
spark.conf.set("spark.sql.files.ignoreCorruptFiles", "true")
 
if os.getenv('CLUSTER')=='PRINCE':
    path_to_data='/scratch/spf248/twitter/data'
else:
    path_to_data='/user/spf248/twitter/data'

# +
df=spark.read.option(
"compression","bzip2").option(
"multiLine","true").option(
"encoding","UTF-8").json(os.path.join(path_to_data,'timelines',this_batch,'API',country_code,'*.json.bz2'))

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

df = df.drop_duplicates(subset=['tweet_id'])
df = df.withColumn('created_at', to_timestamp('created_at',"EEE MMM dd HH:mm:ss ZZZZZ yyyy"))
df = df.withColumn('tweet_longitude', F.col('tweet_coordinates').getItem(0))
df = df.withColumn('tweet_latitude',  F.col('tweet_coordinates').getItem(1))
df = df.drop('tweet_coordinates')
df = df.withColumn('year',year('created_at').cast("string"))
df = df.withColumn('month',month('created_at').cast("string"))
# -

start = timer()
most_recent_date = df.groupby('user_id').agg(F.max('created_at').alias('created_at'))
most_recent_id = df.join(most_recent_date,on=['user_id','created_at']).select('user_id','tweet_id','created_at')
print('USERS:', most_recent_id.count())
most_recent_id.write.mode("overwrite").parquet(os.path.join(path_to_data,'timelines',this_batch,'most_recent_id',country_code))
end = timer()
print('DONE IN', round(end - start), 'SEC')

start = timer()
print('STATUSES:', df.count())
df.write.partitionBy("year", "month").mode("overwrite").format("orc").save(os.path.join(path_to_data,'timelines',this_batch,'extract',country_code))
end = timer()
print('DONE IN', round(end - start), 'SEC')

# Country: MX
# This batch: 062020
# USERS: 770861
# STATUSES: 165868130
# DONE IN 2135 SEC

# Country: BR
# This batch: 062020
# USERS: 893142
# STATUSES: 323572705
# DONE IN 2263 SEC


