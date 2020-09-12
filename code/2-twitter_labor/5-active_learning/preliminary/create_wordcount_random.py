import os
import logging
import argparse
import pandas as pd
import numpy as np
import pytz
import pyarrow
from pathlib import Path
from transformers import BertTokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Import data from random set
    random_data_dir = Path('/scratch/mt4493/twitter_labor/twitter-labor-data/data/random_chunks_with_operations')
    rank = 0
    for parquet_file in random_data_dir.glob('*.parquet'):
        df = pd.read_parquet(parquet_file, columns=['tweet_id', 'text', 'convbert_tokenized_text'])
        df = df.explode('convbert_tokenized_text')
        df = df['convbert_tokenized_text'].value_counts().rename_axis('word').reset_index(name='count')
        df = df.set_index('word')
        if rank == 0:
            wordcount_df = df
        else:
            wordcount_df = wordcount_df.add(df, fill_value=0)
        rank = + 1
    wordcount_df = wordcount_df.reset_index()
    wordcount_df.to_parquet(
        '/scratch/mt4493/twitter_labor/twitter-labor-data/data/wordcount_random/wordcount_random.parquet')
    print(
        "Saved word count to /scratch/mt4493/twitter_labor/twitter-labor-data/data/wordcount_random/wordcount_random.parquet'",
        flush=True)
