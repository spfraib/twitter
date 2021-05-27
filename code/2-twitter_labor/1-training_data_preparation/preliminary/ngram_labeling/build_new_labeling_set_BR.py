from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import Bucketizer, QuantileDiscretizer
from pyspark.sql import Window
from pyspark.sql.types import *
from pyspark.sql.functions import lower, col, lit
import subprocess
import argparse
import os
import unicodedata
import sys

from pyspark.sql.functions import translate, regexp_replace
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf

try:
    spark
except NameError:
    spark = SparkSession.builder.appName("").getOrCreate()

def run_cmd(args_list):
    """
    run linux commands
    """
    # import subprocess
    print('Running system command: {0}'.format(' '.join(args_list)))
    proc = subprocess.Popen(args_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    s_output, s_err = proc.communicate()
    s_return = proc.returncode
    return s_return, s_output, s_err

random_df = spark.read.parquet('/user/mt4493/twitter/random_samples/random_samples_splitted/BR/new_samples')
random_df = random_df.withColumn('text_lowercase', lower(col('text')))

new_ngrams_list = ['me mandaram embora',
                   'primeiro dia de trabalho',
                   'novo (emprego|trampo)']

labelling_path = f'/user/mt4493/twitter/ngram_samples/BR/labeling'
run_cmd(['hdfs', 'dfs', '-mkdir', '-p', labelling_path])
ngram_sample_path = f'/user/mt4493/twitter/ngram_samples/BR/sample_new_1000'
run_cmd(['hdfs', 'dfs', '-mkdir', '-p', labelling_path])
for ngram in new_ngrams_list:
    df_ngram = random_df.filter(random_df.text_lowercase.rlike(ngram))
    share = min(float(1000 / df_ngram.count()), 1.0)
    df_ngram_sample = df_ngram.sample(False, share, seed=0)
    #ngram_str = '_'.join(ngram).replace(' ', '')
    #ngram_folder_name_str = f'{ngram_str}_{df_ngram_sample.count()}'
    #print(ngram_folder_name_str)
    df_ngram_sample = df_ngram_sample.withColumn('ngram', lit(ngram))
    # run_cmd(['hdfs', 'dfs', '-mkdir', '-p', ngram_sample_path])
    df_ngram_sample.write.mode('append').parquet(ngram_sample_path)

    share_150 = min(df_ngram.count(), 150)
    df_ngram_sample = df_ngram_sample.sample(False, 1.0, seed=0)
    df_ngram_sample = df_ngram_sample.limit(share_150)
    df_ngram_sample.write.mode('append').parquet(labelling_path)


