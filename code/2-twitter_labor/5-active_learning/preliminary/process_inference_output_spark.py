from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import Bucketizer, QuantileDiscretizer
from pyspark.sql import Window
from pyspark.sql.types import *
import subprocess
import argparse

try:
    spark
except NameError:
    spark = SparkSession.builder.appName("").getOrCreate()


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_output_folder", type=str,
                        help="Path to the inference data folder.",
                        default="")
    parser.add_argument("--random_chunks_with_operations_folder", type=str,
                        help="Path to folder containing random tweets with operations.",
                        default="/scratch/mt4493/twitter_labor/twitter-labor-data/data/random_chunks_with_operations")
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


# Define base rates
labels = ['is_hired_1mo', 'is_unemployed', 'job_offer', 'job_search', 'lost_job_1mo']
base_rates = [
    1.7342911457049017e-05,
    0.0003534645020523677,
    0.005604641971672389,
    0.00015839552996469054,
    1.455338466552472e-05]
N_random = 92114009
base_ranks = [int(x * N_random) for x in base_rates]
label2rank = dict(zip(labels, base_ranks))

if __name__ == "__main__":
    # Define args from command line
    args = get_args_from_command_line()
    random_tweets_df = spark.read.parquet(args.random_chunks_with_operations_folder)
    for column in labels:
        # read inference data, perform join and isolate top tweets
        inference_df = spark.read.parquet(os.path.join(args.inference_output_folder, column))
        inference_with_text_df = inference_df.join(random_tweets_df, on='tweet_id')
        top_tweets_df = inference_with_text_df.sort(F.col("score").desc()).limit(label2rank[column])
        # prepare paths and save
        output_folder_path = os.path.join(args.inference_output_folder, 'joined', column)
        top_tweets_output_folder_path = os.path.join(output_folder_path, f"top_tweets_{column}")
        run_cmd(['hdfs', 'dfs', '-mkdir', '-p', top_tweets_output_folder_path])
        top_tweets_df.coalesce(1).write.mode("overwrite").parquet(top_tweets_output_folder_path)
        inference_with_text_df.write.mode("overwrite").parquet(output_folder_path)
