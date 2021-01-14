from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import Bucketizer, QuantileDiscretizer
from pyspark.sql import Window
from pyspark.sql.types import *
from pyspark.sql.functions import lower, col, lit
import subprocess
import argparse
import os
import unicodedata
import sys

from pyspark.sql.functions import translate, regexp_replace
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf

try:
    spark
except NameError:
    spark = SparkSession.builder.appName("").getOrCreate()


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_set_path", type=str,
                        help="Path to the folder containing raw tweets.",
                        default="")
    parser.add_argument("--country_code", type=str,
                        help="Path to the inference data folder.",
                        default="US")
    parser.add_argument("--nb_tweets_per_ngram", type=int,
                        help="Number of tweets to sample per ngram.",
                        default=150)
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
    df = spark.read.parquet(args.random_set_path)
    df = df.sample(False, 100000000 / df.count(), seed=0)
    # Keep language-specific tweets
    ngram_dict = {
        'US': [[' i ', 'fired '],
               ['fired me'],
               ['laid off'],
               ['lost my job'],
               ['found ', 'job'],
               ['got ', 'job'],
               ['started', 'job'],
               ['new job'],
               [' i ', 'unemployed'],
               [' i ', 'jobless'],
               [' i ', 'not ', 'working'],
               ['unemployment'],
               ['anyone', 'hiring'],
               ['wish', 'job'],
               ['need', 'job'],
               ['searching', 'job'],
               ['job'],
               ['hiring'],
               ['opportunity'],
               ['apply']],
        'MX': [['me despidieron'],
               ['perdi mi trabajo'],
               ['me corrieron'],
               ['consegui', 'empleo'],
               ['nuevo trabajo'],
               ['nueva chamba'],
               ['encontre', 'trabajo'],
               ['estoy desemplead[o|a]'],
               ['sin empleo'],
               ['sin chamba'],
               [' nini '],
               ['necesito', 'trabajo'],
               ['busco', 'trabajo'],
               ['buscando', 'trabajo'],
               ['alguien', 'trabajo'],
               ['empleo'],
               ['contratando'],
               ['empleo nuevo'],
               ['vacante']],
        'BR': [['perdi', 'emprego'],
               ['perdi', 'trampo'],
               ['fui demitido'],
               ['me demitiram'],
               ['consegui', 'emprego'],
               ['fui contratad[o|a]'],
               ['comeco', 'emprego'],
               ['novo emprego|emprego novo'],
               ['estou desempregad[o|a]'],
               [' eu ', 'sem ', 'emprego'],
               ['gostaria', 'emprego'],
               ['queria', 'emprego'],
               ['preciso', 'emprego'],
               ['procurando', 'emprego'],
               ['enviar', 'curriculo'],
               ['envie', 'curriculo'],
               ['oportunidade', 'emprego'],
               ['temos', 'vagas']]
        }
    ngram_list = ngram_dict[args.country_code]
    for ngram in ngram_list:
        if len(ngram) == 1 and '|' not in ngram[0]:
            df_ngram = df.filter(df.text_lowercase.contains(ngram[0]))
        elif len(ngram) == 1 and '|' in ngram[0]:
            df_ngram = df.filter(df.text_lowercase.rlike(ngram[0]))
        elif len(ngram) == 2:
            regex = f"{ngram[0]}[.\w\s\d]*{ngram[1]}"
            df_ngram = df.filter(df.text_lowercase.rlike(regex))
        elif len(ngram) == 3:
            regex = f"{ngram[0]}[.\w\s\d]*{ngram[1]}[.\w\s\d]*{ngram[2]}"
            df_ngram = df.filter(df.text_lowercase.rlike(regex))
        share = min(float(1000 / df_ngram.count()), 1.0)
        df_ngram_sample = df_ngram.sample(False, share, seed=0)
        ngram_str = '_'.join(ngram).replace(' ', '')
        ngram_folder_name_str = f'{ngram_str}_{df_ngram_sample.count()}'
        print(ngram_folder_name_str)
        ngram_sample_path = f'/user/spf248/twitter/data/ngram_samples/{args.country_code}/sample_1000'
        df_ngram_sample = df_ngram_sample.withColumn('ngram', lit(ngram_str))
        # run_cmd(['hdfs', 'dfs', '-mkdir', '-p', ngram_sample_path])
        df_ngram_sample.write.mode('append').parquet(ngram_sample_path)
    df_ngrams_all_samples = spark.read.parquet(
        f'/user/spf248/twitter/data/ngram_samples/{args.country_code}/sample_1000')
    labelling_path = f'/user/spf248/twitter/data/ngram_samples/{args.country_code}/labeling'
    run_cmd(['hdfs', 'dfs', '-mkdir', '-p', labelling_path])
    f = df_ngrams_all_samples.groupby('ngram').count()
    f = f.withColumn('frac', F.when(col('count') < args.nb_tweets_per_ngram, 1).otherwise(
        args.nb_tweets_per_ngram / col('count')))
    frac_dict = dict(f.select('ngram', 'frac').collect())
    df_sampled = df_ngrams_all_samples.sampleBy('ngram', fractions=frac_dict)
    df_sampled.write.parquet(labelling_path)

