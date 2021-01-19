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

    # necessary
    parser.add_argument("--train_data_path", type=str, help="Path to the training data. Must be in csv format.",
                        default="")
    parser.add_argument("--eval_data_path", type=str, help="Path to the evaluation data. Must be in csv format.",
                        default="")
    parser.add_argument("--holdout_data_path", type=str, help="Path to the holdout data. Must be in csv format.",
                        default=None)
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int)
    parser.add_argument("--model_name", type=str,
                        help="Select the model to use.", default='bert')
    parser.add_argument("--model_type", type=str, default='bert-base-cased')
    parser.add_argument("--output_dir", type=str, help="Define a folder to store the saved models")
    parser.add_argument("--slurm_job_timestamp", type=str, help="Timestamp when job is launched", default="0")
    parser.add_argument("--slurm_job_id", type=str, help="ID of the job that ran training", default="0")
    parser.add_argument("--intra_epoch_evaluation", type=ParseBoolean, help="Whether to do several evaluations per epoch", default=False)
    parser.add_argument("--nb_evaluations_per_epoch", type=int, help="Number of evaluation to perform per epoch", default=10)
    parser.add_argument("--use_cuda", type=int, help="Whether to use cuda", default=1)
    parser.add_argument("--segment", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
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

    for label in ['lost_job_1mo', 'is_hired_1mo', 'is_unemployed', 'job_offer', 'job_search']:
        # Load eval data
        eval_df = pd.read_csv(args.eval_data_path, lineterminator='\n')
        eval_df = eval_df[['tweet_id', 'text', "class"]]
        eval_df.columns = ['tweet_id', 'text', 'labels']
        # Load best model
        best_model_folder = best_model_folders_dict['iter0']['US'][label]
        model_path = os.path.join('/scratch/mt4493/twitter_labor/trained_models', 'US', best_model_folder,
                                  label, 'models', 'best_model')
        train_args = read_json(filename=os.path.join(model_path, 'model_args.json'))
        best_model = ClassificationModel(args.model_name, model_path, args=train_args)

        # EVALUATION ON EVALUATION SET
        result, model_outputs, wrong_predictions = best_model.eval_model(eval_df)
        scores = np.array([softmax(element)[1] for element in model_outputs])
        # Save scores
        eval_df['score'] = scores
        split = best_model_folder.split('_')
        name_val_file = os.path.splitext(os.path.basename(args.eval_data_path))[0]
        slurm_job_id = split[4]
        seed = split[6]
        model_type = f'{split[0]}-{split[1]}'
        path_to_store_eval_scores = os.path.join(os.path.dirname(args.eval_data_path), 'results',
                                                 f'{model_type}_{str(slurm_job_id)}_seed-{str(seed)}',
                                                 f'{name_val_file}_scores.csv')
        if not os.path.exists(os.path.dirname(path_to_store_eval_scores)):
            os.makedirs(os.path.dirname(path_to_store_eval_scores))
        eval_df.to_csv(path_to_store_eval_scores, index=False)
        logging.info("The scores for the evaluation set were saved at {}".format(path_to_store_eval_scores))