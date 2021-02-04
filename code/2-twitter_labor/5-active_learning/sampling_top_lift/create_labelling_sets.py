import pandas as pd
from pathlib import Path
import numpy as np
import os
import argparse

def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str,
                        default="US")
    parser.add_argument("--iter_number", type=str)
    parser.add_argument("--inference_folder", type=str)

    args = parser.parse_args()
    return args

def turn_token_array_to_string(x):
    list_x = list(x)
    return '_'.join(list_x)

if __name__ == '__main__':
    args = get_args_from_command_line()

    for label in ['is_hired_1mo', 'is_unemployed', 'job_search', 'lost_job_1mo']:
        for ngram in ['two_grams', 'three_grams']:
            ngram_path = f'/scratch/mt4493/twitter_labor/twitter-labor-data/data/active_learning/sampling_top_lift/{args.country_code}/{args.inference_folder}/{label}/{ngram}'
            list_parquet = Path(ngram_path).glob('*.parquet')
            df = pd.concat(map(pd.read_parquet, list_parquet)).reset_index(drop=True)
            df['tokens_str'] = df['tokens'].apply(turn_token_array_to_string)
            df_sample = df.groupby('tokens_str', group_keys=False).apply(lambda df: df.sample(5)).reset_index(drop=True)
            df_sample = df_sample[['tweet_id', 'text', 'tokens_str']]
            output_path = f'/scratch/mt4493/twitter_labor/twitter-labor-data/data/active_learning/sampling_top_lift/{args.country_code}/{args.inference_folder}/labeling_sample'
            df_sample.to_parquet(os.path.join(output_path, f'sample_{label}_{ngram}.parquet'), index=False)

    labeling_path = f'/scratch/mt4493/twitter_labor/twitter-labor-data/data/ngram_samples/{args.country_code}/iter{args.iter_number}/labeling'
    list_sample_parquet = Path(output_path).glob('*.parquet')
    df = pd.concat(map(pd.read_parquet, list_sample_parquet)).reset_index(drop=True)
    df.to_parquet(os.path.join(labeling_path, 'merged.parquet'))
