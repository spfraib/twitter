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
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args_from_command_line()
    data_path = '/scratch/mt4493/twitter_labor/twitter-labor-data/data'
    random_path = f'{data_path}/random_samples/random_samples_splitted'
    random_path_new_samples = Path(os.path.join(random_path, args.country_code, 'new_samples'))
    random_df = pd.concat(
        pd.read_parquet(parquet_file)
        for parquet_file in random_path_new_samples.glob('*.parquet')
    )
    base_ranks_dict = {
        'US': {
            'is_hired_1mo': 30338,
            'is_unemployed': 21613,
            'job_offer': 538490,
            'job_search': 47970,
            'lost_job_1mo': 2040}}
    sample_df_list = list()
    for label in ['is_hired_1mo', 'lost_job_1mo', 'job_search', 'is_unemployed', 'job_offer']:
        inference_path = os.path.join(data_path,'inference')
        scores_path = Path(os.path.join(inference_path, args.country_code, args.inference_folder, 'output', label))
        scores_df = pd.concat(
            pd.read_parquet(parquet_file)
            for parquet_file in scores_path.glob('*.parquet')
        )
        all_df = scores_df.merge(random_df, on="tweet_id", how='inner')
        all_df['rank'] = all_df['score'].rank(method='first', ascending=False)
        all_df = all_df.sort_values(by=['rank']).reset_index(drop=True)
        all_df = all_df[:base_ranks_dict[args.country_code][label]]
        sample_df = all_df.sample(n=100).reset_index(drop=True)
        sample_df['label'] = label
        sample_df_list.append(sample_df)
    appended_sample_df = pd.concat(sample_df_list)
    output_path = f'{data_path}/active_learning/sampling_top_lift/{args.country_code}/{args.inference_folder}'
    appended_sample_df.to_parquet(os.path.join(output_path, 'top_tweets.parquet'), index=False)