import pandas as pd
import argparse
import os
from pathlib import Path


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str,
                        help="Country code",
                        default="US")
    parser.add_argument("--inference_folder", type=str)
    parser.add_argument("--new_iteration_folder", type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args_from_command_line()
    data_path = '/scratch/mt4493/twitter_labor/twitter-labor-data/data'
    already_labelled_ids_path = os.path.join(data_path, 'active_learning', 'sampling_top_lift', args.country_code,
                                             args.inference_folder, 'already_labelled_ids.parquet')
    already_labelled_ids_df = pd.read_parquet(already_labelled_ids_path)
    already_labelled_ids_df['tweet_id'] = already_labelled_ids_df['tweet_id'].astype(str)
    already_labelled_labels_path = os.path.join(data_path, 'qualtrics', args.country_code, 'old_iters', 'labeling')
    already_labelled_labels_df = pd.concat([pd.read_parquet(path) for path in Path(already_labelled_labels_path).glob('*.parquet')])
    already_labelled_labels_df['tweet_id'] = already_labelled_labels_df['tweet_id'].astype(str)
    labels_df = already_labelled_ids_df.merge(already_labelled_labels_df, on=['tweet_id']).reset_index(drop=True)

    output_path = os.path.join(data_path, 'train_test', args.country_code, args.new_iteration_folder, 'raw', 'old_labels.parquet')
    labels_df.to_parquet(output_path, index=False)