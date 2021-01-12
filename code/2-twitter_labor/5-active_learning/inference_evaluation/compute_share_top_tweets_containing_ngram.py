import os
import numpy as np
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.functions import lower, col, lit

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

if __name__ == '__main__':
    args = get_args_from_command_line()
    ngram_dict = {
        'US': {
            'lost_job_1mo': [
                "(^|\W)i[ve|'ve| ][\w\s\d]* fired",
                "i got fired",
                "just got fired",
                "laid off",
                "lost my job"
            ],
            'is_hired_1mo': [
                "found[.\w\s\d]*job",
                "(^|\W)just[\w\s\d]* hired",
                "i got hired",
                "got[.\w\s\d]*job",
                "started[.\w\s\d]*job",
                "new job"
            ],
            'is_unemployed': [
                "(^|\W)i[m|'m|ve|'ve| am| have]['\w\s\d]*unemployed",
                "unemployed",
                "(^|\W)i[m|'m|ve|'ve| am| have]['\w\s\d]*jobless",
                "jobless",
                "unemployment"
            ],
            'job_search': [
                "anyone[.\w\s\d]*hiring",
                "wish[.\w\s\d]*job",
                "need[.\w\s\d]*job",
                "searching[.\w\s\d]*job",
                "(^|\W)looking[\w\s\d]* gig[\W]",
                "(^|\W)applying[\w\s\d]* position[\W]",
                "(^|\W)find[\w\s\d]* job[\W]"
            ],
            'job_offer': [
                'job',
                'hiring',
                'opportunity',
                'apply'
            ]
        }}
    all_ngram_list = [item for sublist in list(ngram_dict[args.country_code].values()) for item in sublist]
    share_dict = dict()

    base_rates = {
        'US':
            [5.97e-5, 3.03e-5, 8.82e-4, 6.19e-5, 6.65e-6],
        # past values
        # 6.91e-05,
        # 1.18e-05,
        # 2.28e-03,
        # 3.51e-05,
        # 5.44e-06],
        'MX': [2.73e-05, 1.11e-05, 1.77e-04, 8.36e-06, 1.64e-06],
        'BR': [7.52e-06, 1.51e-05, 1.43e-04, 2.86e-05, 3.6e-06]}

    N_random = {
        'US': 100002226,
        'MX': 99998628,
        'BR': 99984967}
    labels = ['is_hired_1mo', 'is_unemployed', 'job_offer', 'job_search', 'lost_job_1mo']
    base_ranks=[int(x*N_random[args.country_code]) for x in base_rates[args.country_code]]
    label2rank=dict(zip(labels,base_ranks))

    for label in labels:
        path_to_tweets = os.path.join('/user/mt4493/twitter/random_samples/random_samples_splitted', args.country_code,
                                      'evaluation')  # Random set of tweets
        path_to_scores = os.path.join('/user/mt4493/twitter/inference', args.country_code, args.model_folder, 'output',
                                      label)  # Prediction scores from classification
        # path_to_evals = os.path.join('/user/mt4493/twitter/evaluation', args.country_code, args.model_folder,
        #                              label)  # Where to store the sampled tweets to be labeled
        # run_cmd(['hdfs', 'dfs', '-mkdir', '-p', path_to_evals])
        tweets = spark.read.parquet(os.path.join(path_to_tweets))
        scores = spark.read.parquet(os.path.join(path_to_scores))
        df = tweets.select('tweet_id', 'text').join(scores.select('tweet_id', 'score'), on='tweet_id')
        df = df.withColumn('text_lowercase', lower(col('text')))
        df = df.withColumn("rank", F.row_number().over(Window.orderBy(F.desc("score"))))
        df = df.filter(df.rank < label2rank[label])
        df_ngram_class = df.filter(df['text_lowercase'].rlike("|".join(ngram_dict[args.country_code][label])))
        df_ngram_all_class = df.filter(df['text_lowercase'].rlike("|".join(all_ngram_list)))
        share_dict[f'{label}_class'] = df_ngram_class.count() / df.count()
        share_dict[f'{label}_all_class'] = df_ngram_all_class.count() / df.count()
    results = list(map(list, share_dict.items()))
    results_df = spark.createDataFrame(results, ["share_type", "share"])
    results_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(
        os.path.join('/user/mt4493/twitter/inference_evaluation/share_ngram', args.country_code))

