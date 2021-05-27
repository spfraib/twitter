import os
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import argparse
import subprocess

def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str, default='US')
    parser.add_argument("--inference_folder", type=str)

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

if __name__ == '__main__':
    spark = SparkSession.builder.appName("").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    args = get_args_from_command_line()
    path_to_tweets = os.path.join('/user/mt4493/twitter/twitter-labor-data/random_samples/random_samples_splitted',
                                  args.country_code,
                                  'new_samples')  # Random set of tweets
    random_df = spark.read.parquet(path_to_tweets)
    # remove already labelled tweets
    iteration_number = int(args.inference_folder.split('-')[0][-1])
    raw_labels_path_dict = {'US': {0: 'jan5_iter0',
                              1: 'apr19_iter1_adaptive',
                              2: 'apr22_iter2_adaptive',
                              3: 'feb25_iter3'},
                            'MX': {0: 'feb27_iter0', 1: 'mar12_iter1', 2: 'mar23_iter2', 3: 'mar30_iter3'},
                            'BR': {0: 'feb16_iter0', 1: 'mar12_iter1', 2: 'mar24_iter2', 3: 'apr1_iter3'}}
    path_to_labels = f'twitter/twitter-labor-data/train_test/US/{raw_labels_path_dict[args.country_code][iteration_number]}/raw/all_labels_with_text.parquet'
    labels = spark.read.parquet(path_to_labels)
    labels = labels.select('tweet_id')
    random_df = random_df.join(F.broadcast(labels), on='tweet_id', how='left_anti')

    for label in ['is_hired_1mo', 'lost_job_1mo', 'is_unemployed', 'job_search', 'job_offer']:
        path_to_scores = os.path.join('/user/mt4493/twitter/twitter-labor-data/inference', args.country_code, args.inference_folder, 'output',
                                      label)  # Prediction scores from classification
        scores_df = spark.read.parquet(path_to_scores)
        path_to_adaptive_retrieval_sets = os.path.join('/user/mt4493/twitter/twitter-labor-data/adaptive_retrieval_sets', args.country_code,
                                      args.inference_folder, label)  # Prediction scores from classification
        run_cmd(['hdfs', 'dfs', '-mkdir', '-p', path_to_adaptive_retrieval_sets])
        df = random_df.join(scores_df, on='tweet_id', how='inner')
        new_tweets = df.orderBy(F.desc("score")).limit(100)
        new_tweets.coalesce(1).write.parquet(os.path.join(path_to_adaptive_retrieval_sets))
