import os
import sys
import socket
import re
import numpy as np
import string
from timeit import default_timer as timer
from datetime import datetime
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, desc, row_number, col, year, month, dayofmonth, dayofweek, to_timestamp, size, \
    isnan, lower, rand, lit
import pyspark.sql.functions as F
from pyspark.sql.types import MapType, StringType, IntegerType, StructType, StructField, FloatType, ArrayType

try:
    spark
except NameError:
    spark = SparkSession.builder.appName("").getOrCreate()

path_to_data = '/user/spf248/twitter/data'
print('Path to users:', path_to_data)

print('Import')
df = spark.read.option(
    "multiLine", "true").option(
    "encoding", "UTF-8").option(
    "mode", "FAILFAST").json(
    os.path.join(path_to_data, 'users', 'API'))

df.printSchema()

df = df.select('id_str', 'created_at', 'location', 'statuses_count', 'followers_count', 'friends_count',
               'profile_image_url_https', 'name', 'screen_name', 'description', 'lang').drop_duplicates(subset=['id_str'])

# +
print('Save')
start = timer()

df.write.mode("overwrite").json(os.path.join(path_to_data, 'users', 'users-profile'))

end = timer()
print('DONE IN', round(end - start), 'SEC')
