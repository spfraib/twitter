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
from timeit import default_timer as timer
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import MapType, StringType, IntegerType, StructType, StructField, FloatType, ArrayType
import numpy as np

try:
    spark
except NameError:
    if 'samuel' in socket.gethostname().lower():
        print('Create Local SparkSession')
        spark = SparkSession.builder.config(
        "spark.driver.host", "localhost").appName(
        "get-bad-files").getOrCreate()
    else:
        print('Create Cluster SparkSession')
        spark = SparkSession.builder.appName(
        "get-bad-files").getOrCreate()
# -

print('Hostname:', socket.gethostname())
if  'samuel' in socket.gethostname().lower():
    path_to_data='../../data'
else:
    path_to_data='/user/spf248/twitter/data'

# +
print('List files to be processed...')

hadoop = spark.sparkContext._jvm.org.apache.hadoop
fs = hadoop.fs.FileSystem
conf = hadoop.conf.Configuration() 
path = hadoop.fs.Path(os.path.join(path_to_data,'timelines','historical','API','*','*.json.bz2'))
fList = [ str(f.getPath()).replace('hdfs://dumbo','') for f in fs.get(conf).globStatus(path) ]
np.random.seed(0)
fList=np.random.permutation(sorted(fList))

print('# Files:', len(fList))

# +
# fList=[]
# with open('missing_files.txt', 'r') as f:
#     for line in f:
#         fList.append(line)
# fString='\n'.join([x for x in ''.join(fList).split('\n')])
# -

n_chunks=len(fList)
print('# Chunks:', n_chunks)
chunks = np.array_split(fList, n_chunks)


def extract_chunk(i_chunk,chunk):
    try:
        print('Load Chunk:',i_chunk)
        df=spark.read.option(
        "compression","bzip2").option(
        "multiLine","true").option(
        "encoding","UTF-8").json(list(chunk))
    except:
        print('Error with chunk', i_chunk)
        print('\n'.join(list(chunk)))


for i_chunk,chunk in enumerate(chunks):
    start = timer()
    print('Load Chunk:',i_chunk)
    extract_chunk(i_chunk,chunk)
    end = timer()
    print('TIME:', round(end - start), 'SEC')
