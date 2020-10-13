from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import Bucketizer, QuantileDiscretizer
from pyspark.sql import Window
from pyspark.sql.types import *
from pyspark.sql.functions import lower, col
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


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_set_path", type=str,
                        help="Path to the folder containing raw tweets.",
                        default="")
    parser.add_argument("--country_code", type=str,
                        help="Path to the inference data folder.",
                        default="US")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Define args from command line
    args = get_args_from_command_line()
    df = spark.read.parquet(args.raw_tweets_path)
    df = df.sample(False, 100000000/df.count(), seed=0)
    # Keep language-specific tweets
    ngram_dict ={'US': [[' fired '],
                             [' lost ',' job '],
                             [' i ',' not ',' working '],
                             [' laid off '],
                             [' found ', ' job '],
                             [' hired '],
                             [' got ', ' job '],
                             [' started ', ' job '],
                             [' new job '],
                             [' i ', ' unemployed '],
                             [' i ', ' jobless '],
                             [' unemployment '],
                             [' anyone ', ' hiring '],
                             [' job search '],
                             [' wish ', ' hire '],
                             [' need ', ' job '],
                             [' searching ', ' job '],
                             [' job '],
                             [' hiring '],
                             [' apply ']]}
    ngram_list = ngram_dict[args.country_code]
    for ngram in ngram_list:
        if len(ngram) == 1:
            df_ngram = df.filter(df.text_lowercase.contains(ngram[0]))
        elif len(ngram) == 2:
            regex = f"{ngram[0]}.*{ngram[1]}"
            df_ngram = df.filter(df.text_lowercase.rlike(regex))
        elif len(ngram) == 3:
            regex = f"{ngram[0]}.*{ngram[1]}.*{ngram[2]}"
            df_ngram = df.filter(df.text_lowercase.rlike(regex))
        df_ngram_sample = df_ngram.sample(False, 1000/df_ngram.count(), seed=0)
        ngram_sample_path = f'/user/mt4493/twitter/random_samples_ngrams/{args.country_code}'
        df_ngram_sample.coalesce(1).write.mode("overwrite").parquet(ngram_sample_path)
