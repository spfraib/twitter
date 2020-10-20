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
#import emoji

try:
    spark
except NameError:
    spark = SparkSession.builder.appName("").getOrCreate()


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str,
                        help="Path to the inference data folder.",
                        default="US")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args_from_command_line()
    country_language_dict = {
        'US': 'en',
        'MX': 'es',
        'BR': 'pt'}
    language = country_language_dict[args.country_code]
    demojize_udf = udf(demojize, StringType())
    df = spark.read.parquet(f'/user/spf248/twitter/data/timelines/historical/extract/{args.country_code}')
    print('Loaded data')
    # drop RT
    df = df.filter(~df.text.contains('RT '))
    print('Dropped RTs')
    # replace links by [LINK] token and handles by @USER
    df = df.withColumn('text_clean', regexp_replace('text', 'http\S+', 'HTTPURL'))
    df = df.withColumn('text_clean',
                       regexp_replace('text_clean', '(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)', '@USER'))
    print('Replaced links and twitter handles by specific tokens')
    # uncase text
    df = df.withColumn('text_clean_uncased', lower(col('text_clean')))
    print('Lowercase text')
    # drop duplicates
    df = df.drop_duplicates(subset=['text_clean_uncased'])


    # replace emojis
    #def demojize(text):
    #   return emoji.demojize(text, language=language)


    #demojize_udf = udf(demojize, StringType())
    #df = df.withColumn('text_clean_uncased', demojize_udf(col('text_clean_uncased')))
    df = df.select("text_clean_uncased")
    df.write.mode("overwrite").format('text').option("header", "false").mode('append').save(
        f'/user/spf248/twitter/data/pretraining/{args.country_code}/preprocessed/clean_tweets_{args.country_code}.txt')
