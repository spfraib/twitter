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
    parser.add_argument("--seedlist", type=str)

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
    ngram_dict_per_class = {
        'US': {
            'lost_job_1mo': ["(^|\W)i[ve|'ve| ][\w\s\d]* fired", "i got fired",
                             "just got fired", 'laid off',
                             'lost my job'],
            'is_hired_1mo': ['found[.\w\s\d]*job', "(^|\W)just[\w\s\d]* hired", "i got hired", 'got [.\w\s\d]*job',
                             'started[.\w\s\d]*job',
                             'new job'],
            'is_unemployed': ["(^|\W)i[m|'m|ve|'ve| am| have]['\w\s\d]*unemployed", 'unemployed',
                              "(^|\W)i[m|'m|ve|'ve| am| have]['\w\s\d]*jobless", 'jobless', 'unemployment'
                              ],
            'job_search': ['anyone[.\w\s\d]*hiring', 'wish[.\w\s\d]*job', 'need[.\w\s\d]*job',
                           'searching[.\w\s\d]*job', "(^|\W)looking[\w\s\d]* gig[\W]",
                           "(^|\W)applying[\w\s\d]* position[\W]", "(^|\W)find[\w\s\d]* job[\W]"
                           ],
            'job_offer': ['job',
                          'hiring',
                          'opportunity',
                          'apply']
        }
    }
    labels = ['is_hired_1mo', 'lost_job_1mo', 'is_unemployed', 'job_offer', 'job_search']

    regex_list = [re.compile(regex) for regex in ngram_dict[args.country_code]]
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
            positive_df['text_lower'] = positive_df['text'].str.lower()
            if args.seedlist == 'all':
                ngram_list = ngram_dict[args.country_code]
                regex_list = [re.compile(regex) for regex in ngram_list]
            elif args.seedlist == 'class_specific':
                ngram_list = ngram_dict_per_class[args.country_code][label]
                regex_list = [re.compile(regex) for regex in ngram_list]
            positive_df['seedlist'] = positive_df['text_lower'].apply(
                lambda x: regex_match_string(ngram_list=ngram_list, regex_list=regex_list,
                                             mystring=x))
            expansion_rate = 1 - (positive_df.loc[positive_df['seedlist'] == 1].shape[0] / positive_df.shape[0])
            # compute diversity
            positive_tweet_list = positive_df['text'].tolist()
            embeddings = model.encode(positive_tweet_list, convert_to_tensor=True)
            cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)
            diversity_score = (-torch.sum(cosine_scores) / (len(positive_tweet_list) ** 2)).item()
            # store results
            results_dict[inference_folder][label]['expansion_rate'] = expansion_rate
            results_dict[inference_folder][label]['diversity_score'] = diversity_score
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
    results_df.to_csv(os.path.join(output_path, f'expansion_positives_seedlist_{args.seedlist}.csv'), index=False)
