# put random set ou puiser les tweets dans hdfs

# move former data to send to labelling to archive:
# /user/spf248/twitter/data/ngram_samples/{args.country_code}/labeling ->
#  /user/spf248/twitter/data/ngram_samples/{args.country_code}/archive/labeling

# load former data to send to labelling: /user/spf248/twitter/data/ngram_samples/{args.country_code}/archive/labeling
# virer tous les tweets relatifs a des dropped n-grams
# load random set ou puiser des tweets
# prendre 150 tweets par nouveau gram
# join anciens tweets non droppés et nouveau

# output au meme endroit qu'avant: /user/spf248/twitter/data/ngram_samples/{args.country_code}/labeling

# modifier new_labels.pkl: modifier code qui crée training/test set pour virer les tweets issus de n-grams droppés

# créer survey
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

random_df = spark.read.parquet('/user/mt4493/twitter/random_samples/random_samples_splitted/US/new_samples')
random_df = random_df.withColumn('text_lowercase', lower(col('text')))

new_ngrams_list = {'regex': ["(^|\W)i[ve|'ve| ][\w\s\d]* fired",
                            "(^|\W)just[\w\s\d]* hired",
                             "(^|\W)i[m|'m|ve|'ve| am| have]['\w\s\d]*unemployed",
                              "(^|\W)i[m|'m|ve|'ve| am| have]['\w\s\d]*jobless",
                             "(^|\W)looking[\w\s\d]* gig[\W]",
                             "(^|\W)applying[\w\s\d]* position[\W]",
                             "(^|\W)find[\w\s\d]* job[\W]"
                   ],
                   'contains': ["i got fired",
                    "just got fired",
                     "i got hired",
                    "unemployed",
                    "jobless"
                   ]}


for ngram_type, ngram_list in new_ngrams_list.items():
    for ngram in ngram_list:
        if ngram_type == 'regex':
            df_ngram = random_df.filter(random_df.text_lowercase.rlike(ngram))
        elif ngram_type == 'contains':
            df_ngram = random_df.filter(random_df.text_lowercase.contains(ngram))
        share = min(float(1000 / df_ngram.count()), 1.0)
        df_ngram_sample = df_ngram.sample(False, share, seed=0)
        #ngram_str = '_'.join(ngram).replace(' ', '')
        #ngram_folder_name_str = f'{ngram_str}_{df_ngram_sample.count()}'
        #print(ngram_folder_name_str)
        ngram_sample_path = f'/user/mt4493/twitter/ngram_samples/US/sample_new_1000'
        df_ngram_sample = df_ngram_sample.withColumn('ngram', lit(ngram))
        # run_cmd(['hdfs', 'dfs', '-mkdir', '-p', ngram_sample_path])
        df_ngram_sample.write.mode('append').parquet(ngram_sample_path)

df_ngrams_all_samples = spark.read.parquet(f'/user/mt4493/twitter/ngram_samples/US/sample_new_1000')
labelling_path = f'/user/mt4493/twitter/ngram_samples/US/labeling'
# run_cmd(['hdfs', 'dfs', '-mkdir', '-p', labelling_path])
f = df_ngrams_all_samples.groupby('ngram').count()
f = f.withColumn('frac', F.when(col('count') < 150, 1).otherwise(150 / col('count')))
frac_dict = dict(f.select('ngram', 'frac').collect())
df_sampled = df_ngrams_all_samples.sampleBy('ngram', fractions=frac_dict)
df_sampled.write.mode('append').parquet(labelling_path)
