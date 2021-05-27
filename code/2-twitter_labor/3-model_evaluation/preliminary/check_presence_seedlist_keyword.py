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
    parser.add_argument("--threshold", type=float,
                        default=0.95)
    args = parser.parse_args()
    return args

def regex_match_string(ngram_list, regex_list, mystring):
    if any(regex.search(mystring) for regex in regex_list):
        return 1
    elif any(regex in mystring for regex in ngram_list):
        return 1
    else:
        return 0

if __name__ == '__main__':
    args = get_args_from_command_line()
    # define paths
    data_path = '/scratch/mt4493/twitter_labor/twitter-labor-data/data'
    random_path = f'{data_path}/random_samples/random_samples_splitted'
    random_path_evaluation = Path(os.path.join(random_path, args.country_code, 'evaluation'))
    # load random set
    random_df = pd.concat(
        pd.read_parquet(parquet_file)
        for parquet_file in random_path_evaluation.glob('*.parquet')
    )
    logger.info('Loaded random data')
    ngram_dict = {
        'US': ['laid off',
               'lost my job',
               'found[.\w\s\d]*job',
               'got [.\w\s\d]*job',
               'started[.\w\s\d]*job',
               'new job',
               'unemployment',
               'anyone[.\w\s\d]*hiring',
               'wish[.\w\s\d]*job',
               'need[.\w\s\d]*job',
               'searching[.\w\s\d]*job',
               'job',
               'hiring',
               'opportunity',
               'apply', "(^|\W)i[ve|'ve| ][\w\s\d]* fired",
               "(^|\W)just[\w\s\d]* hired",
               "(^|\W)i[m|'m|ve|'ve| am| have]['\w\s\d]*unemployed",
               "(^|\W)i[m|'m|ve|'ve| am| have]['\w\s\d]*jobless",
               "(^|\W)looking[\w\s\d]* gig[\W]",
               "(^|\W)applying[\w\s\d]* position[\W]",
               "(^|\W)find[\w\s\d]* job[\W]",
               "i got fired",
               "just got fired",
               "i got hired",
               "unemployed",
               "jobless"
               ]}
    regex_list = [re.compile(regex) for regex in ngram_dict[args.country_code]]
    random_df['text_lower'] = random_df['text'].str.lower()
    random_df['seedlist_keyword'] = random_df['text_lower'].apply(lambda x: regex_match_string(ngram_list=ngram_dict[args.country_code], regex_list=regex_list, mystring=x))
    random_df = random_df[['tweet_id', 'text_lower', 'seedlist_keyword']]
    output_path = f'/scratch/mt4493/twitter_labor/twitter-labor-data/data/random_samples/random_samples_splitted/{args.country_code}/evaluation_seedlist_keyword'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    random_df.to_parquet(os.path.join(output_path, 'evaluation_seedlist_keyword.parquet'), index=False)