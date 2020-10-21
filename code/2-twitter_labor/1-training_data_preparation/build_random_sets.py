from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import Bucketizer, QuantileDiscretizer
from pyspark.sql import Window
from pyspark.sql.types import *
from pyspark.sql.functions import lower, col, regexp_replace
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
                        help="Country code",
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
    df = df.withColumn('text_no_links', F.regexp_replace('text', 'http\S+', ''))
    df = df.drop_duplicates(subset=['text_no_links']).drop('text_no_links')
    print('Dropped duplicates')
    df = df.filter(~df.text.contains('RT '))
    print('Dropped RTs')
    N_all = df.count()
    print('Total number of tweets:', N_all)
    N_sample = 200000000
    pct_sample = min(N_sample / N_all, 1.0)
    df_random = df.sample(False, pct_sample, seed=0)
    print('Random set to 200M tweets')
    # Add lowercased column
    df_random = df_random.withColumn('text_lowercase', lower(col('text')))
    # Accent replacement for spanish and portuguese
    accent_replacements = [
        ('á|à|ã', 'a'),
        ('é|ê|è', 'e'),
        ('í', 'i'),
        ('ò|ó|õ', 'o'),
        ('ú|ü', 'u'),
        ('ñ', 'n'),
        ('ç', 'c')]
    if args.country_code in ['MX', 'BR']:
        for a, b in accent_replacements:
            df_random = df_random.withColumn('text_lowercase', regexp_replace(col('text_lowercase'), a, b))
    df_random.write.mode("overwrite").parquet(
        f'/user/spf248/twitter/data/random_samples/{args.country_code}')

