import os
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import Window
import subprocess


spark = SparkSession.builder.appName("").getOrCreate()

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

path_data = '/user/spf248/twitter/data/active_learning/US/iter_0-convbert-1122153'

for label in ['lost_job_1mo', 'is_unemployed', 'is_hired_1mo', 'job_search', 'job_offer']:
    scores_path = f'/user/spf248/twitter/data/inference/US/iter_0-convbert-1122153-new_samples/output/{label}'
    scores_df = spark.read.parquet(scores_path)
    ranked = Window.orderBy(desc("score"))
    scores_df = scores_df.withColumn('rank', dense_rank().over(ranked))
    for ngram in ['two_grams', 'three_grams']:
        df = spark.read.parquet(os.path.join(path_data, label, ngram))
        df_with_scores = df.join(scores_df, on='tweet_id')
        output_path = os.path.join(path_data, 'label_with_scores', label, ngram)
        run_cmd(['hdfs', 'dfs', '-mkdir', '-p', output_path])
        df_with_scores.write.mode("overwrite").parquet(output_path)


