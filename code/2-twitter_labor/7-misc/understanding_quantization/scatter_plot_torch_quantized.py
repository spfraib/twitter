import pandas as pd
import argparse
import logging
from pathlib import Path
import os
from simpletransformers.classification import ClassificationModel
import torch
import numpy as np
import re
import json
from scipy.special import softmax
import scipy
import matplotlib.pyplot as plt

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


def read_json(filename: str):
    with open(filename) as f_in:
        return json.load(f_in)


if __name__ == '__main__':
    args = get_args_from_command_line()
    # define paths
    data_path = '/scratch/mt4493/twitter_labor/twitter-labor-data/data'
    random_path = f'{data_path}/random_samples/random_samples_splitted'
    random_path_new_samples = Path(os.path.join(random_path, args.country_code, 'evaluation'))

    inference_folder_dict = {
        'US': ['iter_0-convbert-969622-evaluation', 'iter_1-convbert-3050798-evaluation',
               'iter_2-convbert-3134867-evaluation', 'iter_3-convbert-3174249-evaluation',
               'iter_4-convbert-3297962-evaluation']}

    best_model_folders_dict = {
        'US': {
            'iter0': {
                'lost_job_1mo': 'DeepPavlov_bert-base-cased-conversational_jan5_iter0_928497_SEED_14',
                'is_hired_1mo': 'DeepPavlov_bert-base-cased-conversational_jan5_iter0_928488_SEED_5',
                'is_unemployed': 'DeepPavlov_bert-base-cased-conversational_jan5_iter0_928498_SEED_15',
                'job_offer': 'DeepPavlov_bert-base-cased-conversational_jan5_iter0_928493_SEED_10',
                'job_search': 'DeepPavlov_bert-base-cased-conversational_jan5_iter0_928486_SEED_3'
            },
            'iter1': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_feb22_iter1_3045488_seed-2',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_feb22_iter1_3045493_seed-7',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_feb22_iter1_3045488_seed-2',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_feb22_iter1_3045500_seed-14',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_feb22_iter1_3045501_seed-15'},
            'iter2': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_feb23_iter2_3132744_seed-9',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_feb23_iter2_3132736_seed-1',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_feb23_iter2_3132748_seed-13',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_feb23_iter2_3132740_seed-5',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_feb23_iter2_3132741_seed-6'
            },
            'iter3': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_feb25_iter3_3173734_seed-11',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_feb25_iter3_3173731_seed-8',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_feb25_iter3_3173735_seed-12',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_feb25_iter3_3173725_seed-2',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_feb25_iter3_3173728_seed-5'
            },
            'iter4': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_mar1_iter4_3297481_seed-7',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_mar1_iter4_3297477_seed-3',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_mar1_iter4_3297478_seed-4',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_mar1_iter4_3297477_seed-3',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_mar1_iter4_3297484_seed-10'
            }}}

    # load random set
    random_df = pd.concat(
        pd.read_parquet(parquet_file)
        for parquet_file in random_path_new_samples.glob('*.parquet')
    )
    print('Loaded random data')

    inference_path = os.path.join(data_path, 'inference')
    results_dict = dict()
    to_label_list = list()
    for inference_folder in inference_folder_dict[args.country_code]:
        print(f'**** Inference folder: {inference_folder} ****')
        results_dict[inference_folder] = dict()
        for label in ['is_hired_1mo', 'lost_job_1mo', 'job_search', 'is_unemployed', 'job_offer']:
            print(f'** Class: {label} **')
            scores_path = Path(os.path.join(inference_path, args.country_code, inference_folder, 'output', label))
            scores_df = pd.concat(
                pd.read_parquet(parquet_file)
                for parquet_file in scores_path.glob('*.parquet')
            )
            print('Loaded scores')
            all_df = scores_df.merge(random_df, on="tweet_id", how='inner')
            all_df = all_df[:args.topk]
            # load model
            trained_models_path = '/scratch/mt4493/twitter_labor/trained_models'
            model_iter = int(re.findall('_(\d)-', inference_folder)[0])
            best_model_path = os.path.join(trained_models_path, args.country_code,
                                           best_model_folders_dict[args.country_code][f'iter{model_iter}'], label,
                                           'models/best_model')
            train_args = read_json(filename=os.path.join(best_model_path, 'model_args.json'))
            model = ClassificationModel('bert', best_model_path, args=train_args)
            predictions, raw_outputs = model.predict(all_df['text'].tolist())
            scores = np.array([softmax(element)[1] for element in raw_outputs])
            all_df['pytorch_score'] = scores
            # create rank variables
            all_df['pytorch_score_rank'] = all_df['pytorch_score'].rank(method='dense', ascending=False)
            all_df['quantized_score_rank'] = all_df['score'].rank(method='dense', ascending=False)
            # calculate correlation statistics
            print(f"Spearman correlation (from scipy): {scipy.stats.spearmanr(all_df['pytorch_score'], all_df['score'])}")
            print(f"Spearman correlation (rank + pearson): {scipy.stats.pearsonr(all_df['pytorch_score_rank'], all_df['quantized_score_rank'])}")
            print(f"Kendall Tau: {scipy.stats.kendalltau(all_df['pytorch_score_rank'], all_df['quantized_score_rank'])}")
            # plot top tweets
            all_df.plot.scatter(x='pytorch_score', y='score', c='DarkBlue')
            fig_path = '/scratch/mt4493/twitter_labor/twitter-labor-data/data/fig'
            output_path = os.path.join(fig_path, 'top_tweets_pytorch_quantized_scatter_plots', inference_folder)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            plt.savefig(os.path.join(output_path, f'{label}.png'),bbox_inches='tight', format='png' ,dpi=1200, transparent=False)

