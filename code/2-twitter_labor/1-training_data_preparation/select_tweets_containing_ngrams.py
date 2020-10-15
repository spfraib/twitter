from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import Bucketizer, QuantileDiscretizer
from pyspark.sql import Window
from pyspark.sql.types import *
from pyspark.sql.functions import lower, col
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
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Define args from command line
    args = get_args_from_command_line()
    df = spark.read.parquet(args.raw_tweets_path)
    df = df.sample(False, 100000000/df.count(), seed=0)
    # Keep language-specific tweets
    ngram_dict ={'US': [[' i ', 'fired '],
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
                        ['searching', 'job']
                        ['job'],
                        ['hiring'],
                        ['opportunity'],
                        ['apply']
                        ],
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
                        ['temos', 'vagas']
                        ]
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
        df_ngram_sample = df_ngram.sample(False, 1000/df_ngram.count(), seed=0)
        ngram_sample_path = f'/user/mt4493/twitter/random_samples_ngrams/{args.country_code}'
        df_ngram_sample.coalesce(1).write.mode("overwrite").parquet(ngram_sample_path)
