import os
import numpy as np
from datetime import datetime
import re
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.types import *

spark = SparkSession.builder.config('spark.driver.extraJavaOptions', '-Duser.timezone=UTC').config(
    'spark.executor.extraJavaOptions', '-Duser.timezone=UTC').config('spark.sql.session.timeZone', 'UTC').getOrCreate()


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str)
    parser.add_argument("--model_folder", type=str)
    parser.add_argument("--label", type=str)
    parser.add_argument("--magnitude_order", type=str)
    args = parser.parse_args()
    return args


def get_sampled_indices(min_magnitude=2, top_tweets_threshold=100000000):
    index_list = list(range(1, 11))
    start = 91
    range_value = int((top_tweets_threshold - 9 - start) / pow(10, min_magnitude)) + 1
    for i in range(range_value):
        list_to_add = list(range(start + pow(10, min_magnitude) * i, start + pow(10, min_magnitude) * i + 10))
        index_list += list_to_add
    return index_list


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


country_code = 'US'
model_folder = {'US': 'iter_8-2000-3GB-9032948', 'BR': 'iter_9-2000-3GB-8406630'}
labels = ['is_hired_1mo', 'lost_job_1mo', 'is_unemployed', 'job_search', 'job_offer']
path_to_data = '/user/spf248/twitter/data'
path_to_tweets = os.path.join(path_to_data, 'user_timeline', 'user_timeline_extracts', country_code)
path_to_scores = os.path.join(path_to_data, 'labor', 'tweet_scores', 'BERT', country_code,
                              model_folder[country_code] + '*', 'output')
path_to_evals = os.path.join(path_to_data, 'labor', 'eval_timelines', country_code)

tweets_df = spark.read.parquet(path_to_tweets)
scores_df = spark.read.parquet(path_to_scores)
index_list = get_sampled_indices()

for label in labels:
    path_to_label_eval = os.path.join(path_to_evals, label)
    run_cmd(['hdfs', 'dfs', '-mkdir', '-p', path_to_label_eval])
    w = Window.orderBy(F.desc(label))
    scores_df = scores_df.withColumn(f"{label}_rank", F.dense_rank().over(w))
    tmp = scores_df.filter(scores_df[f"{label}_rank"].isin(index_list)).select('tweet_id', class_).sort(
        scores_df[label].desc()).join(tweets_df, on='tweet_id')
    tmp.coalesce(1).write.mode("overwrite").option("header", "true").parquet(path_to_label_eval)

