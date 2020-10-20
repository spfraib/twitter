from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import Bucketizer, QuantileDiscretizer
from pyspark.sql import Window
from pyspark.sql.types import *
from pyspark.sql.functions import lower, col, lit, translate, regexp_replace, udf
import subprocess
import argparse
import os
import unicodedata
import sys
import emoji

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
    parser.add_argument("--nb_tweets_per_ngram", type=int,
                        help="Number of tweets to sample per ngram.",
                        default=150)
    args = parser.parse_args()
    return args


def demojize(text, language):
    return emoji.demojize(text, language=language)


if __name__ == "__main__":
    args = get_args_from_command_line()
    country_language_dict = {'US': 'en', 'MX': 'es', 'BR': 'pt'}
    language = country_language_dict[args.country_code]
    demojize_udf = udf(demojize, StringType())
    df = spark.read.parquet(args.random_set_path)
    # drop RT
    df = df.filter(~df.text.contains('RT '))
    # replace links by [LINK] token and handles by @USER
    df = df.withColumn('text_clean', regexp_replace('text', 'http\S+', 'HTTPURL'))
    df = df.withColumn('text_clean',
                       regexp_replace('text_clean', '(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)', '@USER'))
    # uncase text
    df = df.withColumn('text_clean_uncased', lower(col('text_clean')))
    # drop duplicates
    df = df.drop_duplicates(subset=['text_clean_uncased'])
    # replace emojis
