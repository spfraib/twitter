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
    path_to_tweets = f'/user/spf248/twitter/data/random_samples/{args.country_code}/'
    df = spark.read.parquet(path_to_tweets)
    print('Data was loaded')
    df = df.select('tweet_id', 'text')
    pct_sample = min(100000000 / df.count(), 1.0)
    df_new_samples = df.sample(False, pct_sample, seed=0)
    df_evaluation = df.join(df_new_samples, df.tweet_id == df_new_samples.tweet_id, "left_outer").where(df_new_samples.tweet_id.isNull()).select([col(c) for c in df.columns])
    path_to_df_new_samples = f'/user/spf248/twitter/data/random_samples/random_samples_splitted/{args.country_code}/new_samples'
    path_to_df_evaluation = f'/user/spf248/twitter/data/random_samples/random_samples_splitted/{args.country_code}/evaluation'
    df_new_samples.write.mode("overwrite").parquet(path_to_df_new_samples)
    df_evaluation.write.mode("overwrite").parquet(path_to_df_evaluation)

