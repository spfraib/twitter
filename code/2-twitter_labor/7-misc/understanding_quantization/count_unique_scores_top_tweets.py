import pandas as pd
import argparse
import logging
from pathlib import Path
import os
import re

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str,
                        default="US")
    parser.add_argument("--topk", type=int, default=10000)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args_from_command_line()
    # define paths
    data_path = '/scratch/mt4493/twitter_labor/twitter-labor-data/data'

    inference_folder_dict = {
        'US': ['iter_0-convbert-969622-evaluation', 'iter_1-convbert-3050798-evaluation',
               'iter_2-convbert-3134867-evaluation', 'iter_3-convbert-3174249-evaluation',
               'iter_4-convbert-3297962-evaluation']}

    inference_path = os.path.join(data_path, 'inference')
    for inference_folder in inference_folder_dict[args.country_code]:
        print(f'**** Inference folder: {inference_folder} ****')
        for label in ['is_hired_1mo', 'lost_job_1mo', 'job_search', 'is_unemployed', 'job_offer']:
            print(f'** Class: {label} **')
            scores_path = Path(os.path.join(inference_path, args.country_code, inference_folder, 'output', label))
            scores_df = pd.concat(
                pd.read_parquet(parquet_file)
                for parquet_file in scores_path.glob('*.parquet')
            )
            print('Loaded scores')
            scores_df = scores_df[:args.topk]
            unique_count = len(scores_df['score'].unique())
            print(f'Unique score count: {unique_count}')
            print(f'Unique score share: {unique_count/args.topk}')