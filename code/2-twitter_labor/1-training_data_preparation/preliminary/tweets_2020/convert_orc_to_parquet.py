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
    path_to_tweets = os.path.join('/user/mt4493/twitter/tweets_2020/orc', country_code)
    output_path= os.path.join('/user/mt4493/twitter/tweets_2020/parquet', country_code)

    df = spark.read.orc(os.path.join(path_to_tweets))

    df.write.mode("overwrite").parquet(output_path)
