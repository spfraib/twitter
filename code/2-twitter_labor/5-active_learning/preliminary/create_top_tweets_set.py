from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import Bucketizer, QuantileDiscretizer
from pyspark.sql import Window
from pyspark.sql.types import *
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
    parser.add_argument("--inference_output_folder", type=str,
                        default="iter_0-convbert-test-48-10-900-1538433-new_samples")
    parser.add_argument("--random_sample", type=str,
                        help="Random sample type.",
                        default="new_samples")
    parser.add_argument("--country_code", type=str,
                        default="US")
    parser.add_argument("--data_folder", type=str,
                        default="jan5_iter0")
    args = parser.parse_args()
    return args


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


if __name__ == "__main__":
    # Define args from command line
    args = get_args_from_command_line()
    # Define base rates
    labels = ['lost_job_1mo', 'is_unemployed', 'job_offer', 'is_hired_1mo', 'job_search']
    base_rates = [
        2.0409187578669242e-05,
        0.0002161342156809927,
        0.00538478625440077,
        0.00030337421446090766,
        0.00047969690430090407]

    N_random = {
        'evaluation': 92114009,
        'new_samples': 99994033}
    base_ranks = [int(x * N_random[args.random_sample]) for x in base_rates]
    label2rank = dict(zip(labels, base_ranks))
    # Load random set
    data_path = '/user/mt4493/twitter/random_samples/random_samples_splitted'
    inference_folder = f'/user/mt4493/twitter/inference/{args.country_code}'
    random_set_path = f'{data_path}/{args.country_code}/{args.random_sample}'
    random_tweets_df = spark.read.parquet(random_set_path)
    if args.random_sample == 'new_samples':
        labels_path = f'/user/mt4493/twitter/{args.data_folder}/{args.country_code}/labels'
        labels_df = spark.read.parquet(labels_path)
        random_tweets_df = random_tweets_df.join(labels_df, ['tweet_id'], "leftanti")
    for column in labels:
        # read inference data, perform join and isolate top tweets
        inference_path = os.path.join(inference_folder, args.inference_output_folder)
        inference_df = spark.read.parquet(os.path.join(args.inference_output_folder, 'output', column))
        inference_with_text_df = inference_df.join(random_tweets_df, on='tweet_id')
        top_tweets_df = inference_with_text_df.sort(F.col("score").desc()).limit(label2rank[column])
        # prepare paths and save
        top_tweets_path = f'/user/mt4493/twitter/inference_evaluation/top_tweets/{args.country_code}/{args.inference_output_folder}'
        top_tweets_column_path = os.path.join(top_tweets_path, f"{column}")
        run_cmd(['hdfs', 'dfs', '-mkdir', '-p', top_tweets_column_path])
        top_tweets_df.coalesce(1).write.mode("overwrite").parquet(top_tweets_column_path)
