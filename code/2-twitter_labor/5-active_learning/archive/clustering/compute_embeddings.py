from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import pickle
import argparse
from pathlib import Path
import logging
import pandas as pd
import os

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str)
    parser.add_argument("--model_folder", type=str)
    parser.add_argument("--model_type", type=str, default='distiluse-base-multilingual-cased-v2')
    parser.add_argument("--output_folder", type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args_from_command_line()
    embedder = SentenceTransformer(args.model_type)
    for label in ['lost_job_1mo', 'is_unemployed', 'is_hired_1mo', 'job_search', 'job_offer']:
        logger.info(f"Load and concat data for label {label}")
        scores_dir = Path(
            f'/scratch/mt4493/twitter_labor/twitter-labor-data/data/inference/{args.country_code}/{args.model_folder}/output/{label}')
        random_set_dir = Path(
            f'/scratch/mt4493/twitter_labor/twitter-labor-data/data/random_samples/random_samples_splitted/{args.country_code}/evaluation')
        scores_df = pd.concat(
            pd.read_parquet(parquet_file)
            for parquet_file in scores_dir.glob('*.parquet')
        )
        logger.info(f'Loaded {str(scores_df.shape[0])} scores')
        text_df = pd.concat(
            pd.read_parquet(parquet_file)
            for parquet_file in random_set_dir.glob('*.parquet')
        )
        logger.info(f'Loaded {str(text_df.shape[0])} tweets')
        df = scores_df.merge(text_df, on="tweet_id", how='inner')
        logger.info(f'Merged scores and text. Merge size: {str(df.shape[0])}')
        df['rank'] = df['score'].rank(method='dense', ascending=False)

        logger.info('Start encoding')
        corpus_embeddings = embedder.encode(df['text'].tolist(), show_progress_bar=True, convert_to_numpy=True)
        logger.info('Done encoding')
        if not os.path.exists(os.path.join(args.output_folder, args.model_type)):
            os.makedirs(os.path.join(args.output_folder, args.model_type))
        output_path = f'{args.output_folder}/{args.model_type}/embeddings-{label}.pkl'
        with open(output_path, "wb") as fOut:
            pickle.dump({
                'sentences': df['text'].tolist(),
                'rank': df['rank'].tolist(),
                'embeddings': corpus_embeddings}, fOut)
        logger.info(f'Embeddings for {label} saved at {output_path}')
