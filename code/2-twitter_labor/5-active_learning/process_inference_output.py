import os
import logging
import argparse
import pandas as pd
import numpy as np
import pytz
import pyarrow
from pathlib import Path

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_output_folder", type=str, help="Path to the inference data. Must be in csv format.",
                        default="")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Define args from command line
    args = get_args_from_command_line()
    # Import inference data
    random_data_dir = Path('/scratch/spf248/twitter/data/classification/US/random')
    full_random_df = pd.concat(pd.read_parquet(parquet_file, columns=['tweet_id', 'text']) for parquet_file in random_data_dir.glob('*.parquet'))
    full_random_df = full_random_df.set_index(['tweet_id'])
    print("Loaded all random data")
    for column in ['is_hired_1mo', 'is_unemployed', 'job_offer', 'job_search', 'lost_job_1mo']:
        inference_data_dir = Path(os.path.join(args.inference_output_folder, column))
        full_inference_df = pd.concat(pd.read_parquet(parquet_file) for parquet_file in inference_data_dir.glob('*.parquet'))
        full_inference_with_text_df = full_inference_df.join(full_random_df)
        output_parquet_path = os.path.join(args.inference_output_folder, "{}_all.parquet".format(column))
        full_inference_with_text_df.to_parquet(output_parquet_path)
        print("All data with text and scores saved at {}".format(output_parquet_path))


