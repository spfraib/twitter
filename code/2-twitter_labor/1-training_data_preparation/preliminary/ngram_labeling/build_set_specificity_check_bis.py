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
import os

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

def clean_ngram(ngram):
    for char in ["(^|\W)", "[m|'m|ve|'ve| am| have]", "['\w\s\d]*", "[\W]", "[ve|'ve| ]", "[\w\s\d]*"]:
        ngram = ngram.replace(char, '')
    ngram = ngram.replace(' ', '_')
    return ngram

if __name__ == "__main__":
    df_sample_1000 = spark.read.parquet('/user/mt4493/twitter/ngram_samples/US/sample_1000')
    df_sample_new_1000 = spark.read.parquet('/user/mt4493/twitter/ngram_samples/US/sample_new_1000')
    df = df_sample_1000.union(df_sample_new_1000)
    # df = df.withColumn('text', regexp_replace('text', '\\', ' '))
    already_labelled_count_dict = {
        # "(^|W)i[m|'m|ve|'ve| am| have]['wsd]*jobless": 30,
        "anyone_hiring": 27,
        "wish_job": 26,
        "(^|\W)i[m|'m|ve|'ve| am| have]['\w\s\d]*unemployed": 25,
        "i got fired": 23,
        "laidoff": 22,
        "(^|\W)looking[\w\s\d]* gig[\W]": 22,
        "(^|\W)applying[\w\s\d]* position[\W]": 21,
        "need_job": 21,
        "(^|\W)find[\w\s\d]* job[\W]": 21,
        "(^|\W)i[ve|'ve| ][\w\s\d]* fired": 21,
        "got_job": 21,
        "apply": 20,
        "started_job": 19,
        "opportunity": 19,
        "just got fired": 19,
        "lostmyjob": 18,
        "(^|\W)just[\w\s\d]* hired": 18,
        "job": 17,
        "unemployment": 17,
        "i got hired": 17,
        "unemployed": 14,
        "jobless": 13,
        "newjob": 13,
        "hiring": 12,
        "found_job": 10,
        "searching_job": 9
    }

    for ngram, count in already_labelled_count_dict.items():
        df_ngram = df.filter(df.ngram == ngram)
        if df_ngram.count() >= count:
            df_ngram = df_ngram.limit(30-count)
        df_ngram = df_ngram.select('tweet_id', 'text', 'ngram')
        output_path = os.path.join('/user/mt4493/twitter/ngram_samples/US/specificity_check', clean_ngram(ngram))
        run_cmd(['hdfs', 'dfs', '-mkdir', '-p', output_path])
        df_ngram.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)


    # # f = df.groupby('ngram').count()
    # # f = f.withColumn('frac', F.when(col('count') < 20, 1).otherwise(20 / col('count')))
    # # frac_dict = dict(f.select('ngram', 'frac').collect())
    # # df_sampled = df.sampleBy('ngram', fractions=frac_dict)
    # df_sampled = df.select('tweet_id', 'text', 'ngram')
    # df_sampled.coalesce(1).write.mode("overwrite").option("header", "true").csv('/user/mt4493/twitter/ngram_samples/US/specificity_check')
