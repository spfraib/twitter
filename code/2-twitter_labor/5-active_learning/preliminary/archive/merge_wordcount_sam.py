import os
import logging
import argparse
import pandas as pd
import numpy as np
import pytz
import pyarrow
from pathlib import Path

if __name__ == "__main__":
    # Import inference data
    random_wordcount_data_dir = Path('/scratch/spf248/twitter/data/classification/US/word_count/random')
    full_random_wordcount_df = pd.concat(pd.read_parquet(parquet_file) for parquet_file in random_wordcount_data_dir.glob('*.parquet'))
    full_random_wordcount_df.to_parquet('/scratch/mt4493/twitter_labor/twitter-labor-data/data/wordcount_random/all_wordcount_sam.parquet')