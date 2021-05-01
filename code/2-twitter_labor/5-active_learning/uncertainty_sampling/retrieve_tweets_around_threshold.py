import pandas as pd
import argparse
import os
from pathlib import Path
import numpy as np 
import pickle 
from scipy import optimize
import logging 

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

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
        logger.info(f'Shape old labels: {df.shape[0]}' )
        df = df[['tweet_id']]
        # df['tweet_id'] = df['tweet_id'].apply(lambda x: str(int(float(x))))
        df = df.drop_duplicates().reset_index(drop=True)
        logger.info(f'Shape old labels (after dropping duplicates): {df.shape[0]}' )
        list_labelled_tweet_ids = df['tweet_id'].tolist()
        to_label_df = to_label_df[~to_label_df['tweet_id'].isin(list_labelled_tweet_ids)].reset_index(drop=True)
        return to_label_df
    else:
        return to_label_df

def func(x, params):
    all_calibrated_scores = [1 / (1 + np.exp(-(param[0] * x + param[1]))) for param in params]
    return np.mean(all_calibrated_scores, axis=0) - 0.5

def calibrate(x, params):
    all_calibrated_scores = [1 / (1 + np.exp(-(param[0] * x + param[1]))) for param in params]
    return np.mean(all_calibrated_scores, axis=0)

if __name__ == '__main__':
    args = get_args_from_command_line()
    data_path = '/scratch/mt4493/twitter_labor/twitter-labor-data/data'
    random_path = f'{data_path}/random_samples/random_samples_splitted'
    random_path_new_samples = Path(os.path.join(random_path, args.country_code, 'new_samples'))
    random_df = pd.concat(
        pd.read_parquet(parquet_file)
        for parquet_file in random_path_new_samples.glob('*.parquet')
    )
    # random_df['tweet_id'] = random_df['tweet_id'].apply(lambda x: str(int(float(x))))
    logger.info('Loaded random data')
    logger.info(f'Shape random data: {random_df.shape[0]}')
    raw_labels_path_dict = {'US': {0: 'jan5_iter0',
                              1: 'apr19_iter1_adaptive',
                              2: 'apr22_iter2_adaptive',
                              3: 'feb25_iter3'},
                            'MX': {0: 'feb27_iter0', 1: 'mar12_iter1', 2: 'mar23_iter2', 3: 'mar30_iter3'},
                            'BR': {0: 'feb16_iter0', 1: 'mar12_iter1', 2: 'mar24_iter2', 3: 'apr1_iter3'}}
    for iteration_number in range(int(args.iteration_number)):
        logger.info(f'Iteration {iteration_number}')
        random_count = random_df.shape[0]
        data_folder_name = raw_labels_path_dict[args.country_code][iteration_number]
        path_to_labelled = f'/scratch/mt4493/twitter_labor/twitter-labor-data/data/train_test/{args.country_code}/{data_folder_name}/raw'
        random_df = discard_already_labelled_tweets(
            path_to_labelled=path_to_labelled,
            to_label_df=random_df)
        # random_df = discard_already_labelled_tweets_csv(
        #     path_to_labelled=path_to_labelled,
        #     to_label_df=random_df)
        logger.info(f'Dropped {str(random_count - random_df.shape[0])} tweets already labelled at iteration {iteration_number}')

    folder_dict = {
        0: [{
            'eval': 'iter_0-convbert-969622-evaluation',
            'new_samples': 'iter_0-convbert-1122153-new_samples'}, 'jan5_iter0'],
        1: [{
            'eval': 'iter_1-convbert-3050798-evaluation',
            'new_samples': 'iter_1-convbert-3062566-new_samples'}, 'feb22_iter1'],
        2: [{
            'eval': 'iter_2-convbert-3134867-evaluation',
            'new_samples': 'iter_2-convbert-3139138-new_samples'}, 'feb23_iter2'],
        3: [{
            'eval': 'iter_3-convbert-3174249-evaluation',
            'new_samples': 'iter_3-convbert-3178321-new_samples'}, 'feb25_iter3'],
        4: [{
            'eval': 'iter_4-convbert-3297962-evaluation',
            'new_samples': 'iter_4-convbert-3308838-new_samples'}, 'mar1_iter4']}

    path_to_params = '/scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/3-model_evaluation/expansion/preliminary'
    params_dict = pickle.load(open(os.path.join(path_to_params, f'calibration_dict_uncertainty_10000_iter{args.iteration_number - 1}.pkl'), 'rb'))
    inference_folder = folder_dict[0][0]['eval']
    
    sample_df_list = list()
    for label in ['is_hired_1mo', 'lost_job_1mo', 'job_search', 'is_unemployed', 'job_offer']:
        logger.info(f'Label: {label}')
        # find root
        params = params_dict[label][inference_folder]['params']
        root = optimize.brentq(func, 0, 1, args=(params))
        logger.info(f'Root: {root}')
        logger.info(f'Calibrated score for root: {calibrate(root, params)}')
        # Load scores
        inference_path = os.path.join(data_path,'inference')
        scores_path = Path(os.path.join(inference_path, args.country_code, args.inference_folder, 'output', label))
        scores_df = pd.concat(
            pd.read_parquet(parquet_file)
            for parquet_file in scores_path.glob('*.parquet')
        )
        logger.info('Loaded scores')
        all_df = scores_df.merge(random_df, on="tweet_id", how='inner')
        logger.info('Merged scores and text')
        # sample 100 tweets around 0.5
        all_df['modified_score'] = all_df['score'] - root
        above_threshold_df = all_df.loc[all_df['modified_score'] > 0].nsmallest(50, 'modified_score')
        below_threshold_df = all_df.loc[all_df['modified_score'] < 0].nlargest(50, 'modified_score')
        sample_df = pd.concat([above_threshold_df, below_threshold_df]).reset_index(drop=True)
        sample_df['label'] = label
        sample_df = sample_df[['tweet_id', 'text', 'label', 'score']]
        sample_df_list.append(sample_df)
    appended_sample_df = pd.concat(sample_df_list)
    output_path = f'{data_path}/active_learning/uncertainty_sampling/{args.country_code}/{args.inference_folder}'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    appended_sample_df.to_parquet(os.path.join(output_path, 'top_tweets.parquet'), index=False)






# for label in ['is_hired_1mo', 'lost_job_1mo', 'is_unemployed', 'job_search', 'job_offer']:
#     path_to_scores = os.path.join('/user/mt4493/twitter/twitter-labor-data/inference', country_code, inference_folder, 'calibrated_output', label)  # Prediction scores from classification
#     scores_df = spark.read.parquet(path_to_scores)
#     path_to_adaptive_retrieval_sets = os.path.join('/user/mt4493/twitter/twitter-labor-data/adaptive_retrieval_sets', country_code, inference_folder)  # Prediction scores from classification
#     df = random_df.join(scores_df, on='tweet_id', how='inner')
#     new_tweets = df.orderBy(F.desc("score")).limit(100)
#     new_tweets = new_tweets.withColumn("class", lit(label))
#     new_tweets.coalesce(1).write.mode('append').parquet(os.path.join(path_to_adaptive_retrieval_sets))