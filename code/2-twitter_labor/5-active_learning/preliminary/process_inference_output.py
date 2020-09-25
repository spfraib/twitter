import os
import logging
import argparse
import pandas as pd
import numpy as np
import pytz
import pyarrow
from pathlib import Path
import time

pd.set_option('display.max_columns', None)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_output_folder", type=str,
                        help="Path to the inference data. Must be in parquet format.",
                        default="")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Define args from command line
    args = get_args_from_command_line()
    # Import inference data
    start = time.time()
    print('starting')
    random_data_dir = Path('/scratch/mt4493/twitter_labor/twitter-labor-data/data/random_chunks_with_operations')
    full_random_df = pd.concat(pd.read_parquet(parquet_file, columns=['tweet_id', 'text', 'convbert_tokenized_text',
                                                                      'lowercased_text', 'tokenized_preprocessed_text'])
                               for parquet_file in random_data_dir.glob('*.parquet'))
    print('read full_random_df', full_random_df.shape, time.time() - start)
    print(full_random_df.head())
    # start = time.time()
    # full_random_df = full_random_df.set_index('tweet_id')
    # print("set index", time.time() - start)

    labels = ['is_hired_1mo', 'is_unemployed', 'job_offer', 'job_search', 'lost_job_1mo']
    for column in labels:
        print('>>>', column)

        start = time.time()
        inference_data_dir = Path(os.path.join(args.inference_output_folder, column))
        full_inference_df = pd.concat(
            pd.read_parquet(parquet_file) for parquet_file in inference_data_dir.glob('*.parquet'))
        print('read all inference', time.time() - start)

        print(full_inference_df.head())

        # start = time.time()
        # full_inference_df = full_inference_df.set_index('tweet_id')
        # print("set index", time.time() - start)

        start = time.time()
        full_inference_with_text_df = full_inference_df.merge(full_random_df,
                                                              left_on='tweet_id',
                                                              right_on='tweet_id'
                                                              )
        print('joined', time.time() - start)

        start = time.time()
        full_inference_with_text_df = full_inference_with_text_df.sort_values(by=["score"],
                                                                              ascending=False).reset_index()
        print('sorted', time.time() - start)
        print(full_inference_with_text_df.head())
        print(full_inference_with_text_df.shape)

        start = time.time()
        output_folder_path = os.path.join(args.inference_output_folder, column, 'joined')
        all_data_path = os.path.join(output_folder_path, "{}_all_sorted_joined.pickle".format(column))

        if not os.path.exists(output_folder_path):
            print('creating save folder as it does not exist', output_folder_path)
            os.makedirs(output_folder_path)

        # save all data
        print('saving')
        #     full_inference_with_text_df.to_parquet(all_data_path)
        #     print("All data with text and scores for label {} saved at {}".format(column, all_data_path), time.time() - start)
        with open(all_data_path, 'wb') as f:
            pickle.dump(full_inference_with_text_df, f)
        print("All data with text and scores for label {} saved at {}".format(column, all_data_path),
              time.time() - start)

        # save sample
        print('sample saving')
        start = time.time()
        sample = full_inference_with_text_df[:500000]
        top_sample_data_path = os.path.join(output_folder_path, "{}_top_sample_sorted.parquet".format(column))

        with open(top_sample_data_path, 'wb') as f:
            pickle.dump(sample, f)
        #     sample.to_parquet(top_sample_data_path)
        print('saved sample', time.time() - start)

        del full_inference_with_text_df

    #     break






    # labels = ['is_hired_1mo', 'is_unemployed', 'job_offer', 'job_search', 'lost_job_1mo']
    # for column in labels:
    #     print('>>>', column)
    #
    #     start = time.time()
    #     inference_data_dir = Path(os.path.join(args.inference_output_folder, column))
    #     full_inference_df = pd.concat(
    #         pd.read_parquet(parquet_file) for parquet_file in inference_data_dir.glob('*.parquet'))
    #     print('read all inference', time.time() - start)
    #
    #     print(full_inference_df.head())
    #
    #     # start = time.time()
    #     # full_inference_df = full_inference_df.set_index('tweet_id')
    #     # print("set index", time.time() - start)
    #
    #     start = time.time()
    #     full_inference_with_text_df = full_inference_df.merge(full_random_df,
    #                                                          left_on = 'tweet_id',
    #                                                          right_on = 'tweet_id'
    #                                                          )
    #     print('joined', time.time() - start)
    #
    #     start = time.time()
    #     full_inference_with_text_df = full_inference_with_text_df.sort_values(by=["score"],
    #                                                                           ascending=False).reset_index()
    #     print('sorted', time.time() - start)
    #     print(full_inference_with_text_df.head())
    #
    #     start = time.time()
    #     output_folder_path = os.path.join(args.inference_output_folder, column)
    #     all_data_path = os.path.join(output_folder_path, "{}_all_sorted.parquet".format(column))
    #
    #     # save all data
    #     print('saving')
    #     full_inference_with_text_df.to_parquet(all_data_path)
    #     print("All data with text and scores for label {} saved at {}".format(column, all_data_path), time.time() - start)
    #
    #     # save sample
    #     print('sample saving')
    #     start = time.time()
    #     full_inference_with_text_df = full_inference_with_text_df[:500000]
    #     top_sample_data_path = os.path.join(output_folder_path, "{}_top_sample_sorted.parquet".format(column))
    #     full_inference_with_text_df.to_parquet(top_sample_data_path)
    #     print('saved sample', time.time() - start)
    #
    #     del full_inference_with_text_df
