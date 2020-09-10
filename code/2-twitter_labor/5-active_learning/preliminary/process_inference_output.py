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
    parser.add_argument("--inference_output_folder", type=str, help="Path to the inference data. Must be in parquet format.",
                        default="")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # calculate base rates
    labels = ['is_hired_1mo', 'is_unemployed', 'job_offer', 'job_search', 'lost_job_1mo']
    base_rates = [
        1.7342911457049017e-05,
        0.0003534645020523677,
        0.005604641971672389,
        0.00015839552996469054,
        1.455338466552472e-05]
    N_random = 92114009
    base_ranks = [int(x * N_random) for x in base_rates]
    label2rank = dict(zip(labels, base_ranks))
    # Define args from command line
    args = get_args_from_command_line()
    # Import inference data
    random_data_dir = Path('/scratch/spf248/twitter/data/classification/US/random')
    full_random_df = pd.concat(pd.read_parquet(parquet_file, columns=['tweet_id', 'text']) for parquet_file in random_data_dir.glob('*.parquet'))
    full_random_df = full_random_df.set_index(['tweet_id'])
    print("Loaded all random data")
    for column in labels:
        print(column)
        inference_data_dir = Path(os.path.join(args.inference_output_folder, column))
        full_inference_df = pd.concat(pd.read_parquet(parquet_file) for parquet_file in inference_data_dir.glob('*.parquet'))
        full_inference_with_text_df = full_inference_df.join(full_random_df)
        output_parquet_path = os.path.join(args.inference_output_folder, column, "{}_all.parquet".format(column))
        full_inference_with_text_df.to_parquet(output_parquet_path)
        top_output_parquet_path = os.path.join(args.inference_output_folder, column, "{}_top.parquet".format(column))
        full_inference_with_text_df = full_inference_with_text_df[:label2rank[column]]
        full_inference_with_text_df.to_parquet(top_output_parquet_path)
        print("All data with text and scores for label {} saved at {}".format(column, output_parquet_path))
        print("Top tweets with text and scores for label {} saved at {}".format(column, top_output_parquet_path))



