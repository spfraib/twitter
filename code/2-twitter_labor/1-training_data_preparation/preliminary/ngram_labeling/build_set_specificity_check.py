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


if __name__ == "__main__":
    df_sample_1000 = spark.read.parquet('/user/mt4493/twitter/ngram_samples/US/sample_1000')
    df_sample_new_1000 = spark.read.parquet('/user/mt4493/twitter/ngram_samples/US/sample_new_1000')
    df = df_sample_1000.union(df_sample_new_1000)
    df = df.withColumn('text', regexp_replace('text', '\\', ' '))
    dropped_ngrams_list = ['i_fired', 'firedme', 'i_unemployed', 'i_jobless', 'i_not_working']
    df = df.filter(~df.ngram.isin(dropped_ngrams_list))
    f = df.groupby('ngram').count()
    f = f.withColumn('frac', F.when(col('count') < 20, 1).otherwise(20 / col('count')))
    frac_dict = dict(f.select('ngram', 'frac').collect())
    df_sampled = df.sampleBy('ngram', fractions=frac_dict)
    df_sampled = df_sampled.select('tweet_id', 'text', 'ngram')
    df_sampled.coalesce(1).write.mode("overwrite").option("header", "true").csv('/user/mt4493/twitter/ngram_samples/US/specificity_check')
