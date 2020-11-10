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
# -

# Country: CO
# STATUSES: 775518930
# USERS: 1431978
# TWEETS: 190026981
# TOTAL STATUSES: 937030323
# DONE IN 2506 SEC

# Country: PK
# STATUSES: 260747819
# USERS: 592576
# TWEETS: 40024334
# TOTAL STATUSES: 294998468
# DONE IN 2652 SEC

# Country: AR
# STATUSES: 1831812575
# USERS: 2117579
# TWEETS: 468879229
# TOTAL STATUSES: 2254224324
# DONE IN 2446 SEC

# Country: MX
# STATUSES: 2012877755
# USERS: 2859873
# TWEETS: 343620495
# TOTAL STATUSES: 2099141219
# DONE IN 2250 SEC

# Country: BR
# STATUSES: 3084272754
# USERS: 3612651
# TWEETS: 890178660
# TOTAL STATUSES: 3900474181
# DONE IN 2380 SEC

# Country: US
# STATUSES: 11162126689
# USERS: 15674127
# TWEETS: 3908098415
# TOTAL STATUSES: 14693195710
# DONE IN 17194 SEC

# Country: FR
# STATUSES: 1011595428
# USERS: 1395265
# TWEETS: 338788179
# TOTAL STATUSES: 1318643551
# DONE IN 2171 SEC

# # Config

country_code = "FR"
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
        
# IgnoreCorruptFiles
spark.conf.set("spark.sql.files.ignoreCorruptFiles", "true")
    
print('Hostname:', socket.gethostname())
if  'samuel' in socket.gethostname().lower():
    path_to_data='../../data'
else:
    path_to_data='/user/spf248/twitter/data'
# -

# # Process Timelines

# +
timelines=spark.read.parquet(os.path.join(path_to_data,'timelines','historical','chunks',country_code,'*/*.parquet'))
print('STATUSES:', timelines.count())

# print("DROP DUPLICATE IDS")
# timelines=timelines.drop_duplicates(subset=['tweet_id'])

# users=timelines.select("user_id").distinct()
# users.cache()

# +
start = timer()

most_recent_date=timelines.groupby('user_id').agg(F.max('created_at').alias('created_at'))
most_recent_id=timelines.join(most_recent_date,on=['user_id','created_at']).select('user_id','tweet_id','created_at')
most_recent_id.write.mode("overwrite").parquet(os.path.join(path_to_data,'timelines','historical','most_recent_id',country_code))
print('USERS:', most_recent_id.count())

end = timer()
print('DONE IN', round(end - start), 'SEC')
# -

# # Join Decahose Tweets

# +
tweets=spark.read.parquet(os.path.join(path_to_data,'tweets/tweets-with-identified-location',country_code))
print('TWEETS:', tweets.count())

# tweets=tweets.join(F.broadcast(users),on='user_id')
# print('TWEETS OF PANEL USERS:',tweets.count())

df=(timelines.unionByName(tweets)).drop_duplicates(subset=['tweet_id'])
print('TOTAL STATUSES:', df.count())

df=df.withColumn('year',year('created_at').cast("string"))
df=df.withColumn('month',month('created_at').cast("string"))

# +
start = timer()

df.write.partitionBy("year", "month").mode("overwrite").format("orc").save(os.path.join(path_to_data,'timelines','historical','extract',country_code))

end = timer()
print('DONE IN', round(end - start), 'SEC')
# -

# Country: US
# Create Cluster SparkSession
# Hostname: compute-1-7.local
# IMPORT
# REPARTITION
# DROP DUPLICATE IDS
# LIST USERS WITH TIMELINES
# STATUSES: 4550792893
# USERS: 5770200
# COUNT VALUES THAT ARE NON-NULL AND NON-NAN
# +----------+----------+----------+----------+-------------+---------+---------------+--------------+
# |  tweet_id|      text|tweet_lang|   user_id|user_location| place_id|tweet_longitude|tweet_latitude|
# +----------+----------+----------+----------+-------------+---------+---------------+--------------+
# |4550792893|4550792891|4550792893|4550792893|   4550792893|159080591|       61041033|      61041033|
# +----------+----------+----------+----------+-------------+---------+---------------+--------------+
#
# TWEETS: 3908098415
# TWEETS OF PANEL USERS: 481753540
# STATUSES: 4885224647
# SAVE
# DONE IN 13672 SEC
# Computing Time: 0.18

# Country: AR
# Create Cluster SparkSession
# Hostname: compute-1-9.local
# IMPORT
# REPARTITION
# DROP DUPLICATE IDS
# LIST USERS WITH TIMELINES
# STATUSES: 2038466535
# USERS: 2353925
# COUNT VALUES THAT ARE NON-NULL AND NON-NAN
# +----------+----------+----------+----------+-------------+--------+---------------+--------------+
# |  tweet_id|      text|tweet_lang|   user_id|user_location|place_id|tweet_longitude|tweet_latitude|
# +----------+----------+----------+----------+-------------+--------+---------------+--------------+
# |2038466535|2038466534|2038466535|2038466535|   2038466535|49475184|       14739186|      14739186|
# +----------+----------+----------+----------+-------------+--------+---------------+--------------+
#
# TWEETS: 468879229
# TWEETS OF PANEL USERS: 194885800
# STATUSES: 2181644169
# SAVE
# DONE IN 817 SEC
#

# Country: CO
# Create Cluster SparkSession
# Hostname: compute-2-5.local
# IMPORT
# REPARTITION
# DROP DUPLICATE IDS
# LIST USERS WITH TIMELINES
# STATUSES: 845684609
# USERS: 1560089
# COUNT VALUES THAT ARE NON-NULL AND NON-NAN
# +---------+---------+----------+---------+-------------+--------+---------------+--------------+
# | tweet_id|     text|tweet_lang|  user_id|user_location|place_id|tweet_longitude|tweet_latitude|
# +---------+---------+----------+---------+-------------+--------+---------------+--------------+
# |845684608|845684607| 845684608|845684608|    845684608|20974654|       11754234|      11754234|
# +---------+---------+----------+---------+-------------+--------+---------------+--------------+
#
# TWEETS: 190026981
# TWEETS OF PANEL USERS: 83965996
# STATUSES: 898531831
# SAVE
# DONE IN 489 SEC

# Country: BR
# Create Cluster SparkSession
# STATUSES: 3340729244
# USERS: 3902061
# COUNT VALUES THAT ARE NON-NULL AND NON-NAN
# +----------+----------+----------+----------+-------------+---------+---------------+--------------+
# |  tweet_id|      text|tweet_lang|   user_id|user_location| place_id|tweet_longitude|tweet_latitude|
# +----------+----------+----------+----------+-------------+---------+---------------+--------------+
# |3340729243|3340729238|3340729243|3340729243|   3340729243|132544972|       44658070|      44658070|
# +----------+----------+----------+----------+-------------+---------+---------------+--------------+
#
# TWEETS: 890178660
# TWEETS OF PANEL USERS: 262424772
# STATUSES: 3523016688
# SAVE
# DONE IN 1310 SEC

# Country: PK
# STATUSES: 263095367
# USERS: 596016
# COUNT VALUES THAT ARE NON-NULL AND NON-NAN
# +---------+---------+----------+---------+-------------+--------+---------------+--------------+
# | tweet_id|     text|tweet_lang|  user_id|user_location|place_id|tweet_longitude|tweet_latitude|
# +---------+---------+----------+---------+-------------+--------+---------------+--------------+
# |263095367|263095367| 263095367|263095367|    263095367| 4200737|         837962|        837962|
# +---------+---------+----------+---------+-------------+--------+---------------+--------------+
#
# TWEETS: 40024334
# TWEETS OF PANEL USERS: 18577578
# STATUSES: 275850653
