from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import Bucketizer, QuantileDiscretizer
from pyspark.sql import Window
from pyspark.sql.types import *
from pyspark.sql.functions import lower, col
import subprocess
import argparse
import os

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

    path_to_tweets = f'/user/mt4493/twitter/all_tweets/{args.country_code}/'
    df = spark.read.parquet(path_to_tweets)
    df = df.select('tweet_id', 'text', 'tweet_lang')
    country_language_dict = {'US': 'en', 'MX': 'es', 'BR': 'pt'}
    df = df.where(df.tweet_lang == country_language_dict[args.country_code]).drop('tweet_lang')
    # drop duplicates and RT
    df = df.drop_duplicates(subset=['text'])
    df = df.filter(~df.text.contains('RT '))
    N_all = df.count()
    if N_all > 200000000:
        share_repr_100M = 100000000/N_all
        df_random_1, df_rest = df.randomSplit(weights=[share_repr_100M, 1 - share_repr_100M])
        df_random_2 = df_rest.sample(False, 100000000/df_rest.count(), seed=0)


