from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import Bucketizer, QuantileDiscretizer
from pyspark.sql import Window
from pyspark.sql.types import *
from pyspark.sql.functions import lower, col
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
    parser.add_argument("--random_set_path", type=str,
                        help="Path to the folder containing raw tweets.",
                        default="")
    parser.add_argument("--country_code", type=str,
                        help="Path to the inference data folder.",
                        default="US")
    parser.add_argument("--language_code", type=str,
                        help="Path to folder containing random tweets with operations.",
                        default="en")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Define args from command line
    args = get_args_from_command_line()
    df = spark.read.parquet(args.raw_tweets_path)
    # Keep language-specific tweets
    df = df.withColumn('text_lowercase', lower(col('text')))
    ngram_dict ={'english': [' fired ',
                             (' lost ',' job '),
                             (' i ',' not ',' working '),
                             ' laid off ',
                             (' found ', ' job '),
                             ' hired ',
                             (' got ', ' job '),
                             (' started ', ' job '),
                             ' new job ',
                             (' i ', ' unemployed '),
                             (' i ', ' jobless '),
                             ' unemployment ',
                             (' anyone ', ' hiring '),
                             ' job search ',
                             (' wish ', ' hire '),
                             (' need ', ' job '),
                             (' searching ', ' job '),
                             ' job ',
                             ' hiring ',
                             ' apply ']}
    for ngram in ngram_list:
        if len(ngram) == 1:
            df_ngram = df.filter(df.text_lowercase.contains(ngram))
        elif len(ngram) == 2:
            regex = f"^(?=.*\b{ngram[0]}\b)(?=.*\b{ngram[1]}\b).*$"
            df_ngram = df.filter(df.text_lowercase.rlike(regex))
        elif len(ngram) == 3:
            regex = f"^(?=.*\b{ngram[0]}\b)(?=.*\b{ngram[1]}\b)(?=.*\b{ngram[2]}\b).*$"
            df_ngram = df.filter(df.text_lowercase.rlike(regex))
