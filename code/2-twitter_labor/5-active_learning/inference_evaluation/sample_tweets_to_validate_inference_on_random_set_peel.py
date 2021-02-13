import os
import numpy as np
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql import Window
import argparse
import subprocess

spark = SparkSession.builder.appName("").getOrCreate()


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str)
    parser.add_argument("--model_folder", type=str)
    args = parser.parse_args()
    return args


def get_sampled_indices(n_sample=10, n_cutoff=6):
    sampled_points = []  # index of scores around which we sample n_sample tweets
    sampled_ranks = []  # ranks of sampled tweets
    for point, rank in enumerate(sorted(set([int(x) for i in range(n_cutoff) for x in np.logspace(i, i + 1, i + 1)]))):
        if not point:
            new_ranks = list(range(rank, rank + n_sample))
        else:
            new_ranks = list(range(rank + 1, rank + n_sample + 1))
        print('Index of sampled point:', point)
        print('Sampled ranks:', new_ranks)
        sampled_points.extend([point] * n_sample)
        sampled_ranks.extend(new_ranks)
    return sampled_points, sampled_ranks

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

if __name__ == '__main__':
    args = get_args_from_command_line()
    for label in ['is_hired_1mo', 'lost_job_1mo', 'is_unemployed', 'job_search', 'job_offer']:
        path_to_tweets = os.path.join('/user/mt4493/twitter/twitter-labor-data/random_samples/random_samples_splitted', args.country_code,
                                      'evaluation')  # Random set of tweets
        path_to_scores = os.path.join('/user/mt4493/twitter/twitter-labor-data/inference', args.country_code, args.model_folder, 'output',
                                      label)  # Prediction scores from classification
        path_to_evals = os.path.join('/user/mt4493/twitter/twitter-labor-data/evaluation', args.country_code, args.model_folder,
                                     label)  # Where to store the sampled tweets to be labeled
        run_cmd(['hdfs', 'dfs', '-mkdir', '-p', path_to_evals])
        sampled_points, sampled_ranks = get_sampled_indices()
        print('# Sampled points:', len(set(sampled_points)))
        print('# Sampled tweets:', len(sampled_ranks))

        tweets = spark.read.parquet(os.path.join(path_to_tweets))
        scores = spark.read.parquet(os.path.join(path_to_scores))
        sampled_indices = spark.createDataFrame(zip(sampled_points, sampled_ranks), schema=['point', 'rank'])

        df = tweets.select('tweet_id', 'text').join(scores.select('tweet_id', 'score'), on='tweet_id')
        df = df.withColumn("rank", F.row_number().over(Window.orderBy(F.desc("score"))))
        df = df.join(sampled_indices, on='rank')

        df.coalesce(1).write.mode("overwrite").option("header", "true").csv(path_to_evals)
