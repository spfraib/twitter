import pandas as pd
from simpletransformers.classification import ClassificationModel
import os
import json
import numpy as np
from scipy.special import softmax


def read_json(filename: str):
    with open(filename) as f_in:
        return json.load(f_in)

if __name__ == '__main__':
    country_code = 'US'
    labels = ['is_hired_1mo', 'lost_job_1mo', 'is_unemployed', 'job_offer', 'job_search']
    best_model_paths_dict = {
        'US': {
            'iter0': {
                'lost_job_1mo': 'DeepPavlov_bert-base-cased-conversational_jan5_iter0_928497_SEED_14',
                'is_hired_1mo': 'DeepPavlov_bert-base-cased-conversational_jan5_iter0_928488_SEED_5',
                'is_unemployed': 'DeepPavlov_bert-base-cased-conversational_jan5_iter0_928498_SEED_15',
                'job_offer': 'DeepPavlov_bert-base-cased-conversational_jan5_iter0_928493_SEED_10',
                'job_search': 'DeepPavlov_bert-base-cased-conversational_jan5_iter0_928486_SEED_3'
                # 'lost_job_1mo': 'vinai_bertweet-base_jan5_iter0_928517_SEED_7',
                # 'is_hired_1mo': 'vinai_bertweet-base_jan5_iter0_928525_SEED_15',
                # 'is_unemployed': 'vinai_bertweet-base_jan5_iter0_928513_SEED_3',
                # 'job_offer': 'vinai_bertweet-base_jan5_iter0_928513_SEED_3',
                # 'job_search': 'vinai_bertweet-base_jan5_iter0_928513_SEED_3'
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
            }, }}
    path_data = '/scratch/mt4493/twitter_labor/twitter-labor-data/data/evaluation_metrics/US/recall/sentences'
    path_models = f'/scratch/mt4493/twitter_labor/trained_models/{country_code}'
    output_path = '/scratch/mt4493/twitter_labor/twitter-labor-data/data/evaluation_metrics/US/recall/scored_sentences'
    for label in labels:
        path_file = os.path.join(path_data, f'{country_code}-{label}.txt')
        df = pd.read_csv(path_file, delimiter="\t", header=None)
        df.columns = ['text']
        for iteration in range(5):
            path_best_model = os.path.join(path_models, best_model_paths_dict[country_code][f'iter{iteration}'][label],
                                           label, 'models/best_model')
            train_args = read_json(filename=os.path.join(path_best_model, 'model_args.json'))
            best_model = ClassificationModel('bert', path_best_model, args=train_args)
            predictions, raw_outputs = best_model.predict(df['text'].tolist())
            scores = np.array([softmax(element)[1] for element in raw_outputs])
            df[f'scores_iter{iteration}'] = scores
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        df.to_csv(os.path.join(output_path, f'{country_code}-{label}.csv'), index=False)

