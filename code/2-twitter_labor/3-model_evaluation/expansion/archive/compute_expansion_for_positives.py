from sentence_transformers import SentenceTransformer, util
import pandas as pd
import os
import torch
import argparse
import re


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str, default='US')
    parser.add_argument("--topk", type=int)
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
    model_dict = {
        'US': 'stsb-roberta-large',
        'MX': 'distiluse-base-multilingual-cased-v2',
        'BR': 'distiluse-base-multilingual-cased-v2'}
    model = SentenceTransformer(model_dict[args.country_code])
    inference_folder_dict = {
        'US': ['iter_0-convbert-969622-evaluation', 'iter_1-convbert-3050798-evaluation',
               'iter_2-convbert-3134867-evaluation', 'iter_3-convbert-3174249-evaluation',
               'iter_4-convbert-3297962-evaluation']}
    data_path = '/home/manuto/Documents/world_bank/bert_twitter_labor/twitter-labor-data/data/active_learning/evaluation_inference'
    output_folder = '/home/manuto/Documents/world_bank/bert_twitter_labor/twitter-labor-data/data'

    labels = ['is_hired_1mo', 'lost_job_1mo', 'is_unemployed', 'job_offer', 'job_search']

    results_dict = dict()
    for inference_folder in inference_folder_dict[args.country_code]:
        print(inference_folder)
        results_dict[inference_folder] = dict()
        for label in labels:
            print(label)
            results_dict[inference_folder][label] = dict()
            final_path = os.path.join(data_path, args.country_code, inference_folder, f'{label}.csv')
            df = pd.read_csv(final_path)
            df = df.loc[df['rank'] < args.topk + 11]
            positive_df = df.loc[df['class'] == 1].reset_index(drop=True)
            # print(f'# positives: {positive_df.shape[0]}')
            # compute expansion
            path_initial_labels = '/home/manuto/Documents/world_bank/bert_twitter_labor/twitter-labor-data/data/train_test/US/jan5_iter0/raw'
            seed_positive_df = pd.read_parquet(os.path.join(path_initial_labels, 'all_labels_with_text.parquet'))
            seed_positive_df = seed_positive_df.loc[seed_positive_df[label] == 'yes'].reset_index(drop=True)
            seed_positive_tweet_list = seed_positive_df['text'].tolist()
            seed_positive_embeddings = model.encode(seed_positive_tweet_list, convert_to_tensor=True)
            positive_tweet_list = positive_df['text'].tolist()
            positive_embeddings = model.encode(positive_tweet_list, convert_to_tensor=True)
            cosine_scores = util.pytorch_cos_sim(positive_embeddings, seed_positive_embeddings)
            expansion_rate = (-torch.sum(cosine_scores) / (len(positive_tweet_list) * len(seed_positive_tweet_list))).item()
            # store results
            results_dict[inference_folder][label]['expansion_rate'] = expansion_rate
    # organize results
    results_df = pd.DataFrame.from_dict(results_dict)
    results_list = list()
    for inference_folder in inference_folder_dict[args.country_code]:
        results_iter_df = results_df[inference_folder].apply(pd.Series)
        iter_number = int(re.findall('iter_(\d)', inference_folder)[0])
        results_iter_df['iter'] = iter_number
        results_list.append(results_iter_df)
    results_df = pd.concat(results_list)
    results_df = results_df.reset_index()
    results_df = results_df.sort_values(by=['index', 'iter']).reset_index(drop=True)
    # save results
    folder_name = f'top_{args.topk}'
    output_path = f'{output_folder}/evaluation_metrics/{args.country_code}/{folder_name}'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    results_df.to_csv(os.path.join(output_path, f'expansion_positives_new.csv'), index=False)
