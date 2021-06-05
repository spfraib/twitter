import pandas as pd
import argparse
import os
from pathlib import Path


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str,
                        help="Country code",
                        default="US")
    parser.add_argument("--inference_folder", type=str)
    parser.add_argument("--iteration_number", type=str)

    args = parser.parse_args()
    return args


def discard_already_labelled_tweets(path_to_labelled, to_label_df):
    parquet_file_list = list(Path(path_to_labelled).glob('*.parquet'))
    if len(parquet_file_list) > 0:
        df = pd.read_parquet(os.path.join(path_to_labelled, 'all_labels_with_text.parquet'))
        print(f'Shape old labels: {df.shape[0]}' )
        df = df[['tweet_id']]
        # df['tweet_id'] = df['tweet_id'].apply(lambda x: str(int(float(x))))
        df = df.drop_duplicates().reset_index(drop=True)
        print(f'Shape old labels (after dropping duplicates): {df.shape[0]}' )
        list_labelled_tweet_ids = df['tweet_id'].tolist()
        to_label_df = to_label_df[~to_label_df['tweet_id'].isin(list_labelled_tweet_ids)].reset_index(drop=True)
        return to_label_df
    else:
        return to_label_df

def discard_already_labelled_tweets_csv(path_to_labelled, to_label_df):
    parquet_file_list = list(Path(path_to_labelled).glob('*.csv'))
    if len(parquet_file_list) > 0:
        df = pd.concat(map(pd.read_csv, parquet_file_list)).reset_index(drop=True)
        print(f'Shape old labels (CSV): {df.shape[0]}' )
        df = df[['tweet_id']]
        df['tweet_id'] = df['tweet_id'].apply(lambda x: str(int(float(x))))
        df = df.drop_duplicates().reset_index(drop=True)
        print(f'Shape old labels (CSV) (after dropping duplicates): {df.shape[0]}' )
        list_labelled_tweet_ids = df['tweet_id'].tolist()
        to_label_df = to_label_df[~to_label_df['tweet_id'].isin(list_labelled_tweet_ids)].reset_index(drop=True)
        return to_label_df
    else:
        return to_label_df

if __name__ == '__main__':
    args = get_args_from_command_line()
    data_path = '/scratch/mt4493/twitter_labor/twitter-labor-data/data'
    random_path = f'{data_path}/random_samples/random_samples_splitted'
    random_path_new_samples = Path(os.path.join(random_path, args.country_code, 'new_samples'))
    random_df = pd.concat(
        pd.read_parquet(parquet_file)
        for parquet_file in random_path_new_samples.glob('*.parquet')
    )
    random_df['tweet_id'] = random_df['tweet_id'].apply(lambda x: str(int(float(x))))
    print('Loaded random data')
    print('Shape random data', random_df.shape)
    raw_labels_path_dict = {'US': {0: 'jan5_iter0',
                              1: 'apr19_iter1_adaptive',
                              2: 'apr22_iter2_adaptive',
                              3: 'apr26_iter3_adaptive',
                              4: 'apr27_iter4_adaptive',
                              5: 'may12_iter5_adaptive',
                              6: 'may28_iter6_adaptive'},
                            'MX': {0: 'feb27_iter0', 1: 'mar12_iter1', 2: 'mar23_iter2', 3: 'mar30_iter3'},
                            'BR': {0: 'feb16_iter0', 1: 'mar12_iter1', 2: 'mar24_iter2', 3: 'apr1_iter3'}}
    for iteration_number in range(int(args.iteration_number)):
        print(f'Iteration {iteration_number}')
        random_count = random_df.shape[0]
        data_folder_name = raw_labels_path_dict[args.country_code][iteration_number]
        path_to_labelled = f'/scratch/mt4493/twitter_labor/twitter-labor-data/data/train_test/{args.country_code}/{data_folder_name}/raw'
        random_df = discard_already_labelled_tweets(
            path_to_labelled=path_to_labelled,
            to_label_df=random_df)
        # random_df = discard_already_labelled_tweets_csv(
        #     path_to_labelled=path_to_labelled,
        #     to_label_df=random_df)
        print(f'Dropped {str(random_count - random_df.shape[0])} tweets already labelled at iteration {iteration_number}')

    sample_df_list = list()
    for label in ['is_hired_1mo', 'lost_job_1mo', 'job_search', 'is_unemployed', 'job_offer']:
        inference_path = os.path.join(data_path,'inference')
        scores_path = Path(os.path.join(inference_path, args.country_code, args.inference_folder, 'output', label))
        scores_df = pd.concat(
            pd.read_parquet(parquet_file)
            for parquet_file in scores_path.glob('*.parquet')
        )
        all_df = scores_df.merge(random_df, on="tweet_id", how='inner')
        all_df = all_df.sort_values(by=['score'], ascending=False).reset_index(drop=True)
        sample_df = all_df[:100]
        sample_df['label'] = label
        sample_df = sample_df[['tweet_id', 'text', 'label']]
        sample_df_list.append(sample_df)
    appended_sample_df = pd.concat(sample_df_list)
    output_path = f'{data_path}/active_learning/adaptive_retrieval/{args.country_code}/{args.inference_folder}'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    appended_sample_df.to_parquet(os.path.join(output_path, 'top_tweets.parquet'), index=False)