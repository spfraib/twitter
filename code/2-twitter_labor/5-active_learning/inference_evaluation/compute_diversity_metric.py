from sentence_transformers import SentenceTransformer, util
import pandas as pd
import os
import torch
import argparse

def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str)
    parser.add_argument("--inference_folder", type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args_from_command_line()
    model_dict = {'US': 'stsb-roberta-large', 'MX': 'distiluse-base-multilingual-cased-v2', 'BR': 'distiluse-base-multilingual-cased-v2'}
    model = SentenceTransformer(model_dict[args.country_code])
    data_path = '/home/manuto/Documents/world_bank/bert_twitter_labor/twitter-labor-data/data/active_learning/evaluation_inference'
    for label in ['is_hired_1mo', 'lost_job_1mo', 'is_unemployed', 'job_offer', 'job_search']:
        final_path = os.path.join(data_path, args.country_code, args.inference_folder, f'{label}.csv')
        df = pd.read_csv(final_path)
        positive_df = df.loc[df['class'] == 1].reset_index(drop=True)
        positive_tweet_list = positive_df['text'].tolist()
        embeddings = model.encode(positive_tweet_list, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)
        diversity_score = (-torch.sum(cosine_scores)/(len(positive_tweet_list)**2)).item()
        print(f'{label}: {diversity_score}')

