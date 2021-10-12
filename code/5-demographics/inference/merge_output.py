import json
import pandas as pd
import logging
import argparse
from pathlib import Path
import ast

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str,
                        default="US")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args_from_command_line()
    logger.info(f'Country code: {args.country_code}')
    # define paths
    inference_dir = f'/scratch/spf248/twitter/data/demographics/inference_results/{args.country_code}'
    err_dir = f'/scratch/spf248/twitter/data/demographics/inference_results/{args.country_code}/err'
    output_dir = f'/scratch/spf248/twitter/data/demographics/inference_results/{args.country_code}/output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # store user_dicts in list
    user_dict_list = list()
    for json_path in Path(inference_dir).glob('*.json'):
        with open(json_path, 'r') as f:
            for line in f:
                try:
                    user_dict = ast.literal_eval(line)
                    user_dict_list.append(user_dict)
                except:
                    continue
    # store data in dataframe and save
    user_df = pd.DataFrame(user_dict_list)
    logger.info(f'# users with demographic inference: {user_df.shape[0]}')
    user_df.to_parquet(os.path.join(output_dir, f'{args.country_code}_demo_inf.parquet'), index=False)
    # get count of non resizable pictures
    err_list = list()
    for json_path in Path(err_dir).glob('*.json'):
        with open(json_path, 'r') as f:
            for line in f:
                try:
                    line = line.replace('\n', '')
                    if line.isdigit():
                        err_list.append(str(line))
                except:
                    continue
    logger.info(f'# users with non resizable imgs: {len(err_list)}')