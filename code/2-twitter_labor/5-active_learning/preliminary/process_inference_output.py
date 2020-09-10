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
        full_inference_with_text_df = full_inference_with_text_df.sort_values(by=["score"], ascending=False).reset_index()
        output_folder_path = os.path.join(args.inference_output_folder,column)
        all_data_path = os.path.join(output_folder_path, "{}_all_sorted.parquet".format(column))
        chunks_data_path = os.path.join(output_folder_path, 'sorted_chunks')
        # save all data
        full_inference_with_text_df.to_parquet(all_data_path)
        print("All data with text and scores for label {} saved at {}".format(column, all_data_path))
        # save sorted chunks
        if not os.path.exists(chunks_data_path):
            os.makedirs(chunks_data_path)
        for chunk_inference_with_text_df in np.array_split(full_inference_with_text_df,1000):
            start_index = chunk_inference_with_text_df.index.start
            end_index = chunk_inference_with_text_df.index.end
            chunk_inference_with_text_df = chunk_inference_with_text_df.reset_index()
            chunk_path = os.path.join(chunks_data_path, '{}_sorted_chunk_{}_{}.parquet'.format(column, str(start_index), str(end_index)))
            chunk_inference_with_text_df.to_parquet(chunk_path)




