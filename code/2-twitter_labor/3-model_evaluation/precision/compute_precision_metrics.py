import pandas as pd
import os
import re
import numpy as np
import argparse

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
    rank_threshold_dict = {
        'is_hired_1mo': {
            0: 95249,
            1: 31756,
            2: 4819,
            3: 23401,
            4: 3834},
        'is_unemployed': {
            0: 2177012,
            1: 4260,
            2: 5719,
            3: 108568,
            4: 26946},
        'job_offer': {
            0: 3101900,
            1: 1235111,
            2: 562596,
            3: 523967,
            4: 1258549},
        'job_search': {
            0: 1501613,
            1: 136421,
            2: 205400,
            3: 36363, 4: 456326},
        'lost_job_1mo': {
            0: 89397,
            1: 1,
            2: 130115,
            3: 11613,
            4: 0}}

    list_folder_dict = {
        'US': ['iter_0-convbert-969622-evaluation', 'iter_1-convbert-3050798-evaluation',
               'iter_2-convbert-3134867-evaluation', 'iter_3-convbert-3174249-evaluation',
               'iter_4-convbert-3297962-evaluation']}
    country_code = args.country_code
    data_path = f'/home/manuto/Documents/world_bank/bert_twitter_labor/twitter-labor-data/data/active_learning/evaluation_inference/{country_code}'
    results_dict = dict()
    for inference_folder in list_folder_dict[country_code]:
        for label in ['is_hired_1mo', 'lost_job_1mo', 'job_search', 'is_unemployed', 'job_offer']:
            if label not in results_dict.keys():
                results_dict[label] = dict()
            csv_path = os.path.join(data_path, inference_folder, f'{label}.csv')
            iter_number = int(re.findall('iter_(\d)', inference_folder)[0])
            df = pd.read_csv(csv_path)
            results_dict[label][f'iter_{iter_number}'] = dict()
            # compute and save metrics
            precision_top20 = df[:20]['class'].value_counts(dropna=False, normalize=True)[1]
            if args.method == 'topk':
                df_top_T = df.loc[df['rank'] < args.topk]
            elif args.method == 'threshold':
                df_top_T = df.loc[df['rank'] < rank_threshold_dict[label][iter_number]]
            if df_top_T.shape[0] < 2:
                precision_top_T = np.nan
            else:
                precision_top_T = df_top_T['class'].value_counts(dropna=False, normalize=True)[1]
            results_dict[label][f'iter_{iter_number}']['precision_top20'] = precision_top20
            results_dict[label][f'iter_{iter_number}']['precision_top_T'] = precision_top_T
    # print(pd.DataFrame.from_dict(results_dict).T)
    # organize results
    results_df = pd.DataFrame.from_dict(results_dict).T
    results_list = list()
    for inference_folder in results_df.columns:
        results_iter_df = results_df[inference_folder].apply(pd.Series)
        iter_number = int(re.findall('iter_(\d)', inference_folder)[0])
        results_iter_df['iter'] = iter_number
        results_list.append(results_iter_df)
        # print(results_iter_df)
    results_df = pd.concat(results_list).reset_index()
    results_df = results_df.sort_values(by=['index', 'iter']).reset_index(drop=True)
    if args.method == 'threshold':
        folder_name = 'threshold_95'
    elif args.method == 'topk':
        folder_name = f'top_{args.topk}'
    output_path = f'/home/manuto/Documents/world_bank/bert_twitter_labor/twitter-labor-data/data/evaluation_metrics/{folder_name}'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    results_df.to_csv(os.path.join(output_path, 'precision_metrics.csv'), index=False)
