import pandas as pd
import argparse
import logging
from pathlib import Path
import os
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
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
    parser.add_argument("--method", type=str)
    parser.add_argument("--threshold", type=float,
                        default=0.95)
    parser.add_argument("--topk", type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args_from_command_line()
    # define paths
    data_path = '/scratch/mt4493/twitter_labor/twitter-labor-data/data'
    random_path = f'{data_path}/random_samples/random_samples_splitted'
    random_path_new_samples = Path(os.path.join(random_path, args.country_code, 'evaluation'))
    # preliminary data
    # tweet_count_seedlist_dict = {
    #     'US': {
    #         'lost_job_1mo': 6749,
    #         'is_hired_1mo': 17818,
    #         'is_unemployed': 20831,
    #         'job_search': 16499,
    #         'job_offer': 512267}}
    # expansion_rate_denom = sum(tweet_count_seedlist_dict[args.country_code].values())
    inference_folder_dict = {
        'US': ['iter_0-convbert-969622-evaluation', 'iter_1-convbert-3050798-evaluation',
               'iter_2-convbert-3134867-evaluation', 'iter_3-convbert-3174249-evaluation',
               'iter_4-convbert-3297962-evaluation']}
    # diversity_model_dict = {
    #     'US': 'stsb-roberta-large',
    #     'MX': 'distiluse-base-multilingual-cased-v2',
    #     'BR': 'distiluse-base-multilingual-cased-v2'}
    # diversity_model = SentenceTransformer(diversity_model_dict[args.country_code])

    # load random set
    random_df = pd.concat(
        pd.read_parquet(parquet_file)
        for parquet_file in random_path_new_samples.glob('*.parquet')
    )
    logger.info('Loaded random data')
    # load seedlist keyword indicator
    seedlist_path = f'/scratch/mt4493/twitter_labor/twitter-labor-data/data/random_samples/random_samples_splitted/{args.country_code}/evaluation_seedlist_keyword'
    seedlist_df = pd.read_parquet(os.path.join(seedlist_path, 'evaluation_seedlist_keyword.parquet'))
    random_df =  random_df.merge(seedlist_df, on="tweet_id", how='inner')

    inference_path = os.path.join(data_path, 'inference')
    results_dict = dict()
    to_label_list = list()
    for inference_folder in inference_folder_dict[args.country_code]:
        logger.info(f'**** Inference folder: {inference_folder} ****')
        results_dict[inference_folder] = dict()
        for label in ['is_hired_1mo', 'lost_job_1mo', 'job_search', 'is_unemployed', 'job_offer']:
            logger.info(f'** Class: {label} **')
            scores_path = Path(os.path.join(inference_path, args.country_code, inference_folder, 'output', label))
            scores_df = pd.concat(
                pd.read_parquet(parquet_file)
                for parquet_file in scores_path.glob('*.parquet')
            )
            logger.info('Loaded scores')
            all_df = scores_df.merge(random_df, on="tweet_id", how='inner')
            all_df = all_df.sort_values(by=['score'], ascending=False).reset_index(drop=True)
            # # keep aside 50 top tweets
            # top_df = all_df[:50]
            # top_df['tweet_type'] = 'top_50'
            # restrict to score > T
            if args.method == 'threshold':
                all_df = all_df.loc[all_df['score'] > args.threshold].reset_index(drop=True)
            elif args.method == 'topk':
                all_df = all_df[:args.topk]
            all_seedlist_df = all_df.loc[all_df['seedlist_keyword'] == 1]
            all_df['inference_folder'] = inference_folder
            logger.info(f'# tweets with score > {args.threshold}: {all_df.shape[0]}')
            # compute and save expansion rate
            results_dict[inference_folder][label] = dict()
            if all_df.shape[0] == 0:
                expansion_rate = np.nan
            else:
                expansion_rate = all_seedlist_df.shape[0] / all_df.shape[0]
            logger.info(f'Expansion rate: {expansion_rate}')
            results_dict[inference_folder][label]['expansion_rate'] = expansion_rate
            # compute and save diversity score
            # if all_df.shape[0] > 0:
            #     tweet_list = all_df['text'].tolist()
            #     embeddings = diversity_model.encode(tweet_list, convert_to_tensor=True)
            #     cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)
            #     diversity_score = (-torch.sum(cosine_scores) / (len(tweet_list) ** 2)).item()
            #     logger.info(f'Diversity score: {diversity_score}')
            #     results_dict[inference_folder][label]['diversity_score'] = diversity_score
            # else:
            #     results_dict[inference_folder][label]['diversity_score'] = np.nan
            # save tweets to label for precision estimate
            # if all_df.shape[0] > 0:
            #     sample_df = all_df.sample(n=50).reset_index(drop=True)
            #     sample_df['tweet_type'] = 'sample_50'
            #     to_label_df = pd.concat([top_df, sample_df]).reset_index(drop=True)
            #     to_label_list.append(to_label_df)
            # else:
            #     logger.info('Only sending top 50 tweets to labeling')
            #     to_label_list.append(top_df)
    # organize results
    results_df = pd.DataFrame.from_dict(results_dict)
    results_list = list()
    for inference_folder in inference_folder_dict[args.country_code]:
        results_iter_df = results_df[inference_folder].apply(pd.Series)
        iter_number = int(re.findall('iter_(\d)', inference_folder)[0])
        results_iter_df['iter'] = iter_number
        results_list.append(results_iter_df)
    results_df = pd.concat(results_list)
    # save results
    if args.method == 'threshold':
        folder_name = f'threshold_{int(args.threshold*100)}'
    elif args.method == 'topk':
        folder_name = f'top_{args.topk}'
    output_path = f'{data_path}/evaluation_metrics/{args.country_code}/{folder_name}'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    results_df = results_df.reset_index()
    results_df = results_df.sort_values(by=['index', 'iter']).reset_index(drop=True)
    results_df.to_csv(os.path.join(output_path, 'expansion.csv'), index=False)
    # appended_to_label_df = pd.concat(to_label_list)
    # output_path = f'{data_path}/evaluation/{args.country_code}/{args.inference_folder}'
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    # appended_to_label_df.to_parquet(os.path.join(output_path, 'top_tweets.parquet'), index=False)
