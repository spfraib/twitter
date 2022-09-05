import pandas as pd
import argparse
from simpletransformers.classification import ClassificationModel
import json
import os
import numpy as np
from scipy.special import softmax
import logging 

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        help="Select the model to use.", default='bert')
    parser.add_argument("--model_type", type=str, default='DeepPavlov/bert-base-cased-conversational')
    args = parser.parse_args()
    return args

def read_json(filename: str):
    with open(filename) as f_in:
        return json.load(f_in)

if __name__ == '__main__':
    args = get_args_from_command_line()

    best_model_folders_dict = {
        'iter0': {
            'US': {
                'lost_job_1mo': 'DeepPavlov_bert-base-cased-conversational_jan5_iter0_928497_SEED_14',
                'is_hired_1mo': 'DeepPavlov_bert-base-cased-conversational_jan5_iter0_928488_SEED_5',
                'is_unemployed': 'DeepPavlov_bert-base-cased-conversational_jan5_iter0_928498_SEED_15',
                'job_offer': 'DeepPavlov_bert-base-cased-conversational_jan5_iter0_928493_SEED_10',
                'job_search': 'DeepPavlov_bert-base-cased-conversational_jan5_iter0_928486_SEED_3'
            }}}
    data_path = '/scratch/mt4493/twitter_labor/twitter-labor-data/data/jan5_iter0/US/train_test'
    for label in ['lost_job_1mo', 'is_hired_1mo', 'is_unemployed', 'job_offer', 'job_search']:
        # Load eval data
        eval_data_path = os.path.join(data_path, f'val_{label}.csv')
        eval_df = pd.read_csv(eval_data_path, lineterminator='\n')
        eval_df = eval_df[['tweet_id', 'text', "class"]]
        eval_df.columns = ['tweet_id', 'text', 'labels']
        # Load best model
        best_model_folder = best_model_folders_dict['iter0']['US'][label]
        model_path = os.path.join('/scratch/mt4493/twitter_labor/trained_models', 'US', best_model_folder,
                                  label, 'models', 'best_model')
        train_args = read_json(filename=os.path.join(model_path, 'model_args.json'))
        best_model = ClassificationModel(args.model_name, model_path, args=train_args)

        # EVALUATION ON EVALUATION SET
        result, model_outputs, wrong_predictions = best_model.eval_model(eval_df[['text', 'labels']])
        scores = np.array([softmax(element)[1] for element in model_outputs])
        # Save scores
        eval_df['score'] = scores

        split = best_model_folder.split('_')
        name_val_file = os.path.splitext(os.path.basename(eval_data_path))[0]
        slurm_job_id = split[4]
        seed = split[6]
        model_type = f'{split[0]}-{split[1]}'
        path_to_store_eval_scores = os.path.join(os.path.dirname(eval_data_path), 'results',
                                                 f'{model_type}_{str(slurm_job_id)}_seed-{str(seed)}',
                                                 f'{name_val_file}_scores.csv')
        # if not os.path.exists(os.path.dirname(path_to_store_eval_scores)):
        #     os.makedirs(os.path.dirname(path_to_store_eval_scores))
        eval_df.to_csv(path_to_store_eval_scores, index=False)
        logging.info("The scores for the evaluation set were saved at {}".format(path_to_store_eval_scores))