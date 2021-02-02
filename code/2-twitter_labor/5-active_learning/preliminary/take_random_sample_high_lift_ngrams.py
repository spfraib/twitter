from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import Bucketizer, QuantileDiscretizer
from pyspark.sql import *
from pyspark.sql.types import *
import subprocess
import argparse
import os
import re
from pathlib import Path
from pyspark.sql.functions import *

try:
    spark
except NameError:
    spark = SparkSession.builder.appName("").getOrCreate()


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_folder", type=str,
                        default="iter_0-convbert-test-48-10-900-1538433-new_samples")
    parser.add_argument("--country_code", type=str,
                        default="US")
    parser.add_argument("--data_folder", type=str, default='jan5_iter0')
    parser.add_argument("--set", type=str)
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


def gen_ngram_str(ngram):
    regex = re.compile('[^a-zA-Z_]')
    return regex.sub('', ngram.replace(')(', '_'))


# def sample_random_tweets_and_save(df, top_ngram_dict):
#     df = df.withColumn('text_lowercase', lower(col('text')))
#     for n in [2, 3]:
#         for label in ['is_hired_1mo', 'lost_job_1mo', 'is_unemployed', 'job_search', 'job_offer']:
#             ngram = top_ngram_dict[n][label]
#             df_ngram = df.filter(df.text_lowercase.rlike(ngram))
#             df_ngram = df_ngram.sample(frac=1, random_state=0)
#             if df_ngram.count() > 10:
#                 df_ngram = df_ngram.limit(10)
#             output_path = f'/user/mt4493/twitter/sample_high_lift_ngrams/{args.inference_folder}/random_set/{label}_{str(n)}gram'
#             run_cmd(['hdfs', 'dfs', '-mkdir', '-p', output_path])
#             df_ngram.coalesce(1).write.mode("overwrite").parquet(output_path)
#
# def sample_top_tweets_and_save(df, top_ngram_dict, label):
#     for n in [2, 3]:
#         ngram = top_ngram_dict[n][label]
#         df_ngram = df.filter(df.text_lowercase.rlike(ngram))
#         df_ngram = df_ngram.sample(frac=1, random_state=0)
#         if df_ngram.count() > 10:
#             df_ngram = df_ngram.limit(10)
#         output_path = f'/user/mt4493/twitter/sample_high_lift_ngrams/{args.inference_folder}/top_tweets/{label}_{str(n)}gram'
#         run_cmd(['hdfs', 'dfs', '-mkdir', '-p', output_path])
#         df_ngram.coalesce(1).write.mode("overwrite").parquet(output_path)

if __name__ == '__main__':
    # Define args from command line
    args = get_args_from_command_line()
    labor_data_path = '/scratch/mt4493/bert_twitter_labor/twitter-labor-data/data'
    data_path = f'{labor_data_path}/top_tweets/US/{args.inference_folder}'
    output_path = f'{labor_data_path}/evaluation_inference/clustering/US/{args.inference_folder}'
    # Load random set
    random_folder_path = '/user/mt4493/twitter/random_samples/random_samples_splitted'
    random_sample = 'new_samples'
    random_set_path = f'{random_folder_path}/{args.country_code}/{random_sample}'
    random_tweets_df = spark.read.parquet(random_set_path)
    # Discard already labelled tweets
    labels_path = f'/user/mt4493/twitter/{args.data_folder}/{args.country_code}/labels'
    labels_df = spark.read.parquet(labels_path)
    random_tweets_df = random_tweets_df.join(labels_df, ['tweet_id'], "leftanti")
    # Lowercase text
    random_tweets_df = random_tweets_df.withColumn('text_lowercase', lower(col('text')))
    random_tweets_df.cache()
    random_tweets_df.count()
    # Define top lift ngrams
    top_ngram_dict = {
        3: {
            'lost_job_1mo': ['^(?=.*\bgot\b)(?=.*\bjust\b)(?=.*\bkicked\b).*$',
                             '^(?=.*\bfriend\b)(?=.*\blost\b)(?=.*\btoday\b).*$',
                             '^(?=.*\bmy\b)(?=.*\bphone\b)(?=.*\byesterday\b).*$',
                             '^(?=.*\bfucked\b)(?=.*\blast\b)(?=.*\bnight\b).*$',
                             '^(?=.*\bfor\b)(?=.*\bpulled\b)(?=.*\btime\b).*$'
                             ],
            'job_search': ['^(?=.*\bfind\b)(?=.*\bneed\b)(?=.*\bsomething\b).*$',
                           '^(?=.*\bi\b)(?=.*\bnew\b)(?=.*\bsuggestions\b).*$',
                           '^(?=.*\b?\b)(?=.*\bany\b)(?=.*\brecommendations\b).*$',
                           '^(?=.*\bsoon\b)(?=.*\bto\b)(?=.*\btrip\b).*$',
                           '^(?=.*\banyone\b)(?=.*\bknow\b)(?=.*\bwatch\b).*$'
                           ],
            'job_offer': ['^(?=.*\bbuyer\b)(?=.*\blooking\b)(?=.*\breal\b).*$',
                          '^(?=.*\bestate\b)(?=.*\bon\b)(?=.*\bwe\b).*$',
                          '^(?=.*\blake\b)(?=.*\bsalt\b)(?=.*\but\b).*$',
                          '^(?=.*\ba\b)(?=.*\banalyst\b)(?=.*\bfor\b).*$',
                          '^(?=.*\bin\b)(?=.*\bis\b)(?=.*\bseeking\b).*$'
                          ],
            'is_unemployed': ['^(?=.*\bi\b)(?=.*\blost\b)(?=.*\bvoice\b).*$',
                              '^(?=.*\bam\b)(?=.*\blosing\b)(?=.*\bmy\b).*$',
                              '^(?=.*\ba\b)(?=.*\bbreakdown\b)(?=.*\bhaving\b).*$',
                              '^(?=.*\bhave\b)(?=.*\bheadache\b)(?=.*\bnow\b).*$',
                              '^(?=.*\beat\b)(?=.*\bstarving\b)(?=.*\bto\b).*$'
                              ],
            'is_hired_1mo': ['^(?=.*\bfinally\b)(?=.*\bgot\b)(?=.*\bphone\b).*$',
                             '^(?=.*\bback\b)(?=.*\bmy\b)(?=.*\byay\b).*$',
                             '^(?=.*\bfirst\b)(?=.*\btomorrow\b)(?=.*\bwork\b).*$',
                             '^(?=.*\bkeys\b)(?=.*\bnew\b)(?=.*\bthe\b).*$',
                             '^(?=.*\b!\b)(?=.*\bi\b)(?=.*\binternship\b).*$'
                             ]},
        2: {
            'lost_job_1mo': ['^(?=.*\blost\b)(?=.*\bpower\b).*$',
                             '^(?=.*\bjust\b)(?=.*\bkicked\b).*$',
                             '^(?=.*\bbed\b)(?=.*\bfell\b).*$',
                             '^(?=.*\bbanned\b)(?=.*\bgot\b).*$',
                             '^(?=.*\bblacked\b)(?=.*\bi\b).*$'
                             ],
            'job_search': ['^(?=.*\bnew\b)(?=.*\bsuggestions\b).*$',
                           '^(?=.*\bany\b)(?=.*\brecommendations\b).*$',
                           '^(?=.*\bhobby\b)(?=.*\bneed\b).*$',
                           '^(?=.*\bideas\b)(?=.*\bsomething\b).*$',
                           '^(?=.*\basap\b)(?=.*\bfind\b).*$'
                           ],
            'job_offer': ['^(?=.*\bbuyer\b)(?=.*\blooking\b).*$',
                          '^(?=.*\bam\b)(?=.*\bestate\b).*$',
                          '^(?=.*\bcity\b)(?=.*\but\b).*$',
                          '^(?=.*\bfl\b)(?=.*\breal\b).*$',
                          '^(?=.*\bengineer\b)(?=.*\bsenior\b).*$'
                          ],
            'is_unemployed': ['^(?=.*\bim\b)(?=.*\blosing\b).*$',
                              '^(?=.*\blost\b)(?=.*\bvoice\b).*$',
                              '^(?=.*\bhave\b)(?=.*\bmigraine\b).*$',
                              '^(?=.*\bam\b)(?=.*\bheartless\b).*$',
                              '^(?=.*\battack\b)(?=.*\bhaving\b).*$'
                              ],
            'is_hired_1mo': ['^(?=.*\bgot\b)(?=.*\blicense\b).*$',
                             '^(?=.*\bearly\b)(?=.*\byay\b).*$',
                             '^(?=.*\bmy\b)(?=.*\bpermit\b).*$',
                             '^(?=.*\bcar\b)(?=.*\bfinally\b).*$',
                             '^(?=.*\bfriday\b)(?=.*\bpaid\b).*$'
                             ], }}
    if args.set == 'random_set':
        # Load random set
        random_folder_path = '/user/mt4493/twitter/random_samples/random_samples_splitted'
        random_sample = 'new_samples'
        random_set_path = f'{random_folder_path}/{args.country_code}/{random_sample}'
        random_tweets_df = spark.read.parquet(random_set_path)
        # Discard already labelled tweets
        labels_path = f'/user/mt4493/twitter/{args.data_folder}/{args.country_code}/labels'
        labels_df = spark.read.parquet(labels_path)
        random_tweets_df = random_tweets_df.join(labels_df, ['tweet_id'], "leftanti")
        # Lowercase text
        random_tweets_df.cache()
        random_tweets_df.count()
        random_tweets_df = random_tweets_df.withColumn('text_lowercase', lower(col('text')))
        for label in ['is_hired_1mo', 'lost_job_1mo', 'is_unemployed', 'job_search', 'job_offer']:
            for n in [2, 3]:
                ngram_list = top_ngram_dict[n][label]
                for ngram in ngram_list:
                    ngram_str = gen_ngram_str(ngram)
                    df_ngram = random_tweets_df.filter(random_tweets_df.text_lowercase.rlike(ngram))
                    df_ngram = df_ngram.sample(False, 1.0)
                    if df_ngram.count() > 10:
                        df_ngram = df_ngram.limit(10)
                    output_path = f'/user/mt4493/twitter/sample_high_lift_ngrams/{args.inference_folder}/random_set/{label}/{str(n)}-gram/{ngram_str}'
                    run_cmd(['hdfs', 'dfs', '-mkdir', '-p', output_path])
                    df_ngram = df_ngram.withColumn('ngram', ngram_str)
                    df_ngram.coalesce(1).write.mode("overwrite").parquet(output_path)

    elif args.set == 'top_tweets':
        top_tweets_path = f'/user/mt4493/twitter/inference_evaluation/top_tweets/{args.country_code}/{args.inference_folder}'
        for label in ['is_hired_1mo', 'lost_job_1mo', 'is_unemployed', 'job_search', 'job_offer']:
            top_tweets_df = spark.read.parquet(os.path.join(top_tweets_path, label))
            top_tweets_df = top_tweets_df.withColumn('text_lowercase', lower(col('text')))
            for n in [2, 3]:
                ngram_list = top_ngram_dict[n][label]
                for ngram in ngram_list:
                    ngram_str = gen_ngram_str(ngram)
                    df_ngram = top_tweets_df.filter(top_tweets_df.text_lowercase.rlike(ngram))
                    df_ngram = df_ngram.sample(False, 1.0)
                    if df_ngram.count() > 10:
                        df_ngram = df_ngram.limit(10)
                    output_path = f'/user/mt4493/twitter/sample_high_lift_ngrams/{args.inference_folder}/top_tweets/{label}/{str(n)}-gram/{ngram_str}'
                    run_cmd(['hdfs', 'dfs', '-mkdir', '-p', output_path])
                    df_ngram = df_ngram.withColumn('ngram', ngram_str)
                    df_ngram.coalesce(1).write.mode("overwrite").parquet(output_path)
