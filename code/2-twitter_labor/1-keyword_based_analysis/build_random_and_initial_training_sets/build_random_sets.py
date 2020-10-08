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
    parser.add_argument("--country_code", type=str,
                        help="Path to the inference data folder.",
                        default="US")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Define args from command line
    args = get_args_from_command_line()
    print('Start running code')
    path_to_tweets = f'/user/spf248/twitter/data/timelines/historical/extract/{args.country_code}/'
    df = spark.read.orc(path_to_tweets)
    print('Data was loaded')
    df = df.select('tweet_id', 'text', 'tweet_lang')
    country_language_dict = {'US': 'en', 'MX': 'es', 'BR': 'pt'}
    df = df.where(df.tweet_lang == country_language_dict[args.country_code]).drop('tweet_lang')
    print(f'Selected tweets in {country_language_dict[args.country_code]}')
    # drop duplicates and RT
    df = df.drop_duplicates(subset=['text'])
    print('Dropped duplicates')
    df = df.filter(~df.text.contains('RT '))
    print('Dropped RTs')
    N_all = df.count()
    print('Total number of tweets:', N_all)
    if N_all > 200000000:
        df_random_1, df_random_2 = df.randomSplit(weights=[0.5, 0.5])
        print('Performed random split')
        df_random_1 = df_random_1.limit(100000000)
        print('Limited first random set to 100M tweets')
        df_random_2 = df_random_2.limit(100000000)
        print('Limited second random set to 100M tweets')
        df_random_1.coalesce(1000).write.mode("overwrite").parquet(f'/user/spf248/twitter/data/random_samples/{args.country_code}/random_1')
        print('Outputted first random set')
        df_random_2.coalesce(1000).write.mode("overwrite").parquet(f'/user/spf248/twitter/data/random_samples/{args.country_code}/random_2')
        print('Outputted second random set')

