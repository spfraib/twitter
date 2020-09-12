import os
import logging
import argparse
import pandas as pd
import numpy as np
import pytz
import pyarrow
from pathlib import Path
from transformers import BertTokenizerFast
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from nltk.util import skipgrams
from glob import glob
import getpass


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_env_var(varname, default):
    if os.environ.get(varname) is not None:
        var = int(os.environ.get(varname))
        print(varname, ':', var)
    else:
        var = default
        print(varname, ':', var, '(Default)')
    return var


def k_skip_n_grams(sent, k, n):
    return list(skipgrams(sent, k=k, n=n))


tokenizer = BertTokenizerFast.from_pretrained('DeepPavlov/bert-base-cased-conversational')
text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
               'time', 'url', 'date', 'number'],
    # terms that will be annotated
    annotate={"hashtag", "allcaps", "elongated", "repeated",
              'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens

    # corpus from which the word statistics are going to be used
    # for word segmentation
    segmenter="twitter",

    # corpus from which the word statistics are going to be used
    # for spell correction
    corrector="twitter",

    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words

    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,

    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons])

if __name__ == "__main__":
    # Define args from command line
    random_folder_path = '/scratch/spf248/twitter/data/classification/US/random'
    # Get env vars
    SLURM_ARRAY_TASK_ID = get_env_var('SLURM_ARRAY_TASK_ID', 0)
    SLURM_ARRAY_TASK_COUNT = get_env_var('SLURM_ARRAY_TASK_COUNT', 1)
    SLURM_JOB_ID = get_env_var('SLURM_JOB_ID', 1)
    # Define paths and load data
    random_chunks_paths = list(
        np.array_split(glob(os.path.join(random_folder_path, '*.parquet')), SLURM_ARRAY_TASK_COUNT)[
            SLURM_ARRAY_TASK_ID])
    chunk_df = pd.DataFrame()
    for file in random_chunks_paths:
        chunk_df = pd.concat([chunk_df, pd.read_parquet(file)[['tweet_id', 'text']]])
    # Drop RT
    chunk_df = chunk_df[~chunk_df.text.str.contains("RT", na=False)].reset_index(drop=True)
    # BERT tokenization
    chunk_df['convbert_tokenized_text'] = chunk_df['text'].apply(tokenizer.tokenize)
    # Lowercase
    chunk_df['lowercased_text'] = chunk_df['text'].str.lower()
    # Preprocess
    chunk_df['tokenized_preprocessed_text'] = chunk_df['text'].apply(text_processor.pre_process_doc)
    # Compute all k-skip-n-grams
    #chunk['skipgrams'] = chunk_df['tokenized_preprocessed_text'].apply(k_skip_n_grams, k=2,
    #                                                                   n=3)
    # Save to parquet
    output_path = os.path.join('/scratch/mt4493/twitter_labor/twitter-labor-data/data/random_chunks_with_operations',
                               '{}_random-{}.parquet'.format(str(getpass.getuser()), str(SLURM_ARRAY_TASK_ID)))
    chunk_df.to_parquet(output_path)
