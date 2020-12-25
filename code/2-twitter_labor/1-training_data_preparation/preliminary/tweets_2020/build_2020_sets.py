import os
import numpy as np
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.functions import to_timestamp, year
import argparse
spark = SparkSession.builder.appName("").getOrCreate()

def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str,
                        help="Country code",
                        default="MX")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Define args from command line
    args = get_args_from_command_line()

    country_code=args.country_code
    path_to_tweets = os.path.join('/user/spf248/twitter/data/timelines/decahose/update/062020/extract', country_code)
    output_path= os.path.join('/user/spf248/twitter/tweets_2020', country_code)

    df = spark.read.json(os.path.join(path_to_tweets))

    df = df.select('text', 'created_at', to_timestamp(df.created_at, 'yyyy-MM-dd HH:mm:ss').alias('date'))
    df = df.withColumn('year', year('date'))
    df = df.filter(df.year == 2020)

    df.write.mode("overwrite").parquet(output_path)





