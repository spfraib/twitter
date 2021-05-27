"""
This script launches the training of a Transformer-based text classification model and saves the trained model
in a designated folder.

The train and evaluation input data needs to be in CSV format and contain at least two columns (named 'text' and 'class').

By default, we use BERT base and Conversational BERT.
The default model specs are listed here: https://github.com/ThilinaRajapakse/simpletransformers#args-explained


How to use the script in the command line:
python3 train_bert.py
    --train_data_path <TRAIN_DATA_PATH> \
    --eval_data_path <EVAL_DATA_PATH> \
    --num_labels <NUM_LABELS> \
    --model_name <MODEL_NAME> \
    --model_type <MODEL_TYPE> \
    --output_dir <OUTPUT_DIR> \
Where:
<TRAIN_DATA_PATH>: Path to the training data CSV (required)

<EVAL_DATA_PATH>: Path to the evaluation data CSV (required)

<NUM_LABELS>: Number of classes in the classification problem (default=2)


<MODEL_NAME>: Name of the model to use (required)

<MODEL_TYPE>: Type of model to be used (required)
A list can be found here: https://github.com/ThilinaRajapakse/simpletransformers#current-pretrained-models)

<OUTPUT_DIR>: Path to the folder where the trained models will be stored.


Example usage:
python train_bert.py \
--train_data_path twitter-labor-data/data/jun_27_iter1/BERT/new_train_test_split_with_data_iter0/train_is_hired_1mo.csv \
--eval_data_path twitter-labor-data/data/jun_27_iter1/BERT/new_train_test_split_with_data_iter0/val_is_hired_1mo.csv \
--output_dir trained_bert-base-cased_is_hired_1mo_jun_27_iter1 \
--num_labels 2 \
--num_train_epochs 20
--model_name bert \
--model_type bert-base-cased
"""
import os
import shutil, errno
import logging
import argparse
from simpletransformers.classification import ClassificationModel
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn import metrics
import sklearn
from scipy.special import softmax
import numpy as np
from datetime import datetime
import time
import pytz
import json

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

def ParseBoolean (b):
    if len(b) < 1:
        raise ValueError ('Cannot parse empty string into boolean.')
    b = b[0].lower()
    if b == 't' or b == 'y' or b == '1':
        return True
    if b == 'f' or b == 'n' or b == '0':
        return False
    raise ValueError ('Cannot parse string into boolean.')

def verify_data_format(df: pd.DataFrame) -> None:
    """Verify that the df has the right columns."""
    if not {'text', 'labels'}.issubset(df.columns):
        raise ValueError('Both columns "text" and "labels" are required')


def verify_column_type(df: pd.DataFrame) -> None:
    """Verify that the columns have the right type."""
    if not is_string_dtype(df['text']):
        raise ValueError(
            'The "text" column should be of string type')
    elif not is_numeric_dtype(df['labels']):
        raise ValueError(
            'The "labels" column should be of integer type')


def ensure_output_dir_is_dir(output_dir: str) -> str:
    """Make sure the output_dir ends with a slash."""
    if output_dir[-1] != '/':
        output_dir = f"{output_dir}/"
    return output_dir


def check_command_line_args(args: dict) -> dict:
    """Verify command line arguments have the right format. Update them where necessary."""
    args.output_dir = ensure_output_dir_is_dir(args.output_dir)
    return args


def prepare_filepath_for_storing_model(output_dir: str) -> str:
    """Prepare the filepath where the trained model will be stored.

    :param output_dir: Directory where to store outputs (trained models).
    :return: path_to_store_model: Path where to store the trained model.
    """
    path_to_store_model = os.path.join(output_dir, 'models')
    if not os.path.exists(path_to_store_model):
        os.makedirs(path_to_store_model)
    return path_to_store_model


def prepare_filepath_for_storing_best_model(path_to_store_model: str) -> str:
    path_to_store_best_model = os.path.join(path_to_store_model, 'best_model')
    if not os.path.exists(path_to_store_best_model):
        os.makedirs(path_to_store_best_model)
    return path_to_store_best_model


def convert_score_to_predictions(score):
    if score > 0.5:
        return 1
    elif score <= 0.5:
        return 0

def copy_folder(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise

def read_json(filename: str):
    with open(filename) as f_in:
        return json.load(f_in)

if __name__ == "__main__":
    # Define args from command line
    args = get_args_from_command_line()
    args = check_command_line_args(args)

    # Display the arguments
    logger.info(args)
    # Import data
    train_df = pd.read_csv(args.train_data_path, lineterminator='\n')
    eval_df = pd.read_csv(args.eval_data_path, lineterminator='\n')
    text_column = 'text_segment' if args.segment == 1 else 'text'
    if args.holdout_data_path:
        holdout_df = pd.read_csv(args.holdout_data_path, lineterminator='\n')
        holdout_df = holdout_df[[text_column, 'class']]
        holdout_df.columns = ['text', 'labels']
        verify_data_format(holdout_df)

    # Reformat the data
    train_df = train_df[['tweet_id', text_column, "class"]]
    eval_df = eval_df[['tweet_id', text_column, "class"]]
    train_df.columns = ['tweet_id', 'text', 'labels']
    eval_df.columns = ['tweet_id', 'text', 'labels']

    print("********** Train shape: ", train_df.shape[0], " **********")
    print("********** Eval shape: ", eval_df.shape[0], " **********")

    # Make sure the DataFrame contains the necessary columns
    verify_data_format(train_df)
    verify_data_format(eval_df)

    # Make sure the columns have the right type
    verify_column_type(train_df)
    verify_column_type(eval_df)

    # Prepare paths
    path_to_store_model = prepare_filepath_for_storing_model(output_dir=args.output_dir)
    path_to_store_best_model = prepare_filepath_for_storing_best_model(path_to_store_model)
    # Whether to use_cuda
    if args.use_cuda == 1:
        use_cuda = True
    elif args.use_cuda == 0:
        use_cuda = False
    # Create a ClassificationModel
    ## Define arguments
    name_val_file = os.path.splitext(os.path.basename(args.eval_data_path))[0]
    classification_args = {'train_batch_size': 8, 'overwrite_output_dir': True, 'evaluate_during_training': True,
                                      'save_model_every_epoch': True, 'save_eval_checkpoints': True,
                                      'output_dir': path_to_store_model, 'best_model_dir': path_to_store_best_model,
                                      'evaluate_during_training_verbose': True,
                                      'num_train_epochs': args.num_train_epochs, "use_early_stopping": True,
                                      "early_stopping_delta": 0, "early_stopping_metric": "auroc",
                                      "early_stopping_metric_minimize": False, "tensorboard_dir": f"runs/{args.slurm_job_id}_{name_val_file.replace('val_', '')}/" ,
                                      "manual_seed": args.seed}
    ## Allow for several evaluations per epoch
    if args.intra_epoch_evaluation:
        nb_steps_per_epoch = (train_df.shape[0] // classification_args['train_batch_size']) + 1
        classification_args['evaluate_during_training_steps'] = int(nb_steps_per_epoch // args.nb_evaluations_per_epoch)
        classification_args['early_stopping_patience'] = args.nb_evaluations_per_epoch + 1
    else:
        classification_args['early_stopping_patience'] = 3
    ## Define the model
    model = ClassificationModel(args.model_name, args.model_type, num_labels=args.num_labels, use_cuda=use_cuda,
                                args=classification_args)
    # Define evaluation metrics
    eval_metrics = {
        "precision": sklearn.metrics.precision_score,
        "recall": sklearn.metrics.recall_score,
        "f1": sklearn.metrics.f1_score
    }
    # Train the model
    model.train_model(train_df, eval_df=eval_df, output_dir=path_to_store_model, **eval_metrics)
    logging.info("The training of the model is done")

    if args.intra_epoch_evaluation:
        # Find best model (in terms of evaluation loss) at or after the first epoch
        training_progress_scores_df = pd.read_csv(os.path.join(path_to_store_model, 'training_progress_scores.csv'))
        overall_best_model_step = training_progress_scores_df['global_step'][training_progress_scores_df[['eval_loss']].idxmin()[0]]
        print('Nb steps per epoch: ', nb_steps_per_epoch)
        print('Overall best model step: ', overall_best_model_step)
        if int(overall_best_model_step) >= int(nb_steps_per_epoch):
            print("The best model is found at {} steps, therefore after the first epoch ({} steps).".format(overall_best_model_step, nb_steps_per_epoch))
        else:
            training_progress_scores_after_first_epoch_df = training_progress_scores_df[training_progress_scores_df['global_step'] >= nb_steps_per_epoch]
            best_model_after_first_epoch_step = training_progress_scores_after_first_epoch_df['global_step'][training_progress_scores_after_first_epoch_df[['eval_loss']].idxmin()[0]]
            ## Rename past best_model folder to best_model_overall
            if not os.path.isdir(path_to_store_best_model):
                print("There is no {} folder".format(path_to_store_best_model))
            os.rename(path_to_store_best_model, os.path.join(os.path.dirname(path_to_store_best_model), 'overall_best_model'))
            ## Copy folder of best model at or after first epoch to best_model folder
            if int(best_model_after_first_epoch_step) % int(nb_steps_per_epoch) == 0:
                epoch_number = int(best_model_after_first_epoch_step / nb_steps_per_epoch)
                best_model_after_first_epoch_path = os.path.join(path_to_store_model, 'checkpoint-{}-epoch-{}'.format(str(best_model_after_first_epoch_step), str(epoch_number)))
            else:
                best_model_after_first_epoch_path = os.path.join(path_to_store_model, 'checkpoint-{}'.format(str(best_model_after_first_epoch_step)))
            copy_folder(best_model_after_first_epoch_path, path_to_store_best_model)
            print("The best model is found at {} steps, therefore before the first epoch ({} steps).".format(overall_best_model_step, nb_steps_per_epoch))
            print("The best model at or after the first epoch is found at {} steps.".format(best_model_after_first_epoch_step))
            print("The {} folder is copied at {} and the former best_model folder is renamed overall_best_model.".format(best_model_after_first_epoch_path, path_to_store_best_model))

    # Load best model (in terms of evaluation loss)
    train_args = read_json(filename=os.path.join(path_to_store_best_model, 'model_args.json'))
    best_model = ClassificationModel(args.model_name, path_to_store_best_model, args=train_args)

    # EVALUATION ON EVALUATION SET
    result, model_outputs, wrong_predictions = best_model.eval_model(eval_df[['text', 'labels']])
    scores = np.array([softmax(element)[1] for element in model_outputs])
    y_pred = np.vectorize(convert_score_to_predictions)(scores)
    # Compute AUC
    fpr, tpr, thresholds = metrics.roc_curve(eval_df['labels'], scores)
    auc_eval = metrics.auc(fpr, tpr)
    # Centralize evaluation results in a dictionary
    slurm_job_timestamp = args.slurm_job_timestamp
    slurm_job_id = args.slurm_job_id
    eval_results_eval_set_dict = {'slurm_job_id': slurm_job_id,
                                  'slurm_job_timestamp': slurm_job_timestamp,
                                  'slurm_job_Berlin_date_time': str(datetime.fromtimestamp(int(slurm_job_timestamp),
                                                                                           tz=pytz.timezone(
                                                                                               'Europe/Berlin'))),
                                  'model_type': args.model_type,
                                  'evaluation_data_path': args.eval_data_path,
                                  'precision': metrics.precision_score(eval_df['labels'], y_pred),
                                  'recall': metrics.recall_score(eval_df['labels'], y_pred),
                                  'f1': metrics.f1_score(eval_df['labels'], y_pred),
                                  'auc': auc_eval
                                  }
    # Save evaluation results on eval set
    segmented_str = 'segmented' if args.segment == 1 else 'not_segmented'
    seed_str = f'seed-{args.seed}'
    if "/" in args.model_type:
        args.model_type = args.model_type.replace('/', '-')
    path_to_store_eval_results = os.path.join(os.path.dirname(args.eval_data_path), 'results',
                                              f'{args.model_type}_{str(slurm_job_id)}_{seed_str}',
                                              f'{name_val_file}_evaluation.csv')
    if not os.path.exists(os.path.dirname(path_to_store_eval_results)):
        os.makedirs(os.path.dirname(path_to_store_eval_results))
    pd.DataFrame.from_dict(eval_results_eval_set_dict, orient='index', columns=['value']).to_csv(
        path_to_store_eval_results)
    logging.info(
        "The evaluation on the evaluation set is done. The results were saved at {}".format(path_to_store_eval_results))
    # Save scores
    eval_df['score'] = scores
    path_to_store_eval_scores = os.path.join(os.path.dirname(args.eval_data_path), 'results',
                                              f'{args.model_type}_{str(slurm_job_id)}_{seed_str}',
                                              f'{name_val_file}_scores.csv')
    if not os.path.exists(os.path.dirname(path_to_store_eval_scores)):
        os.makedirs(os.path.dirname(path_to_store_eval_scores))
    eval_df.to_csv(path_to_store_eval_scores, index=False)
    logging.info("The scores for the evaluation set were saved at {}".format(path_to_store_eval_scores))

    # EVALUATION ON HOLDOUT SET
    if args.holdout_data_path:
        result, model_outputs, wrong_predictions = best_model.eval_model(holdout_df[['tweet_id', 'labels']])
        scores = np.array([softmax(element)[1] for element in model_outputs])
        y_pred = np.vectorize(convert_score_to_predictions)(scores)
        # Compute AUC
        fpr, tpr, thresholds = metrics.roc_curve(holdout_df['labels'], scores)
        auc_holdout = metrics.auc(fpr, tpr)
        # Centralize evaluation results in a dictionary
        eval_results_holdout_set_dict = {'slurm_job_id': slurm_job_id,
                                         'slurm_job_timestamp': slurm_job_timestamp,
                                         'slurm_job_Berlin_date_time': str(
                                             datetime.fromtimestamp(int(slurm_job_timestamp),
                                                                    tz=pytz.timezone(
                                                                        'Europe/Berlin'))),
                                         'model_type': args.model_type,
                                         'holdout_data_path': args.holdout_data_path,
                                         'precision': metrics.precision_score(holdout_df['labels'], y_pred),
                                         'recall': metrics.recall_score(holdout_df['labels'], y_pred),
                                         'f1': metrics.f1_score(holdout_df['labels'], y_pred),
                                         'auc': auc_holdout
                                         }
        # Save evaluation results on holdout set
        name_holdout_file = os.path.splitext(os.path.basename(args.holdout_data_path))[0]
        path_to_store_holdout_results = os.path.join(os.path.dirname(args.eval_data_path), 'results',
                                              f'{args.model_type}_{str(slurm_job_id)}_{seed_str}',
                                              f'{name_holdout_file}_evaluation.csv')
        if not os.path.exists(os.path.dirname(path_to_store_holdout_results)):
            os.makedirs(os.path.dirname(path_to_store_holdout_results))
        pd.DataFrame.from_dict(eval_results_holdout_set_dict, orient='index', columns=['value']).to_csv(
            path_to_store_holdout_results)
        logging.info(
            "The evaluation on the holdout set is done. The results were saved at {}".format(
                path_to_store_holdout_results))
        # Save scores
        holdout_df['{}_scores'.format(args.model_type)] = scores
        path_to_store_holdout_scores = os.path.join(os.path.dirname(args.eval_data_path), 'results',
                                              f'{args.model_type}_{str(slurm_job_id)}_{seed_str}',
                                              f'{name_holdout_file}_scores.csv')
        if not os.path.exists(os.path.dirname(path_to_store_holdout_scores)):
            os.makedirs(os.path.dirname(path_to_store_holdout_scores))
        holdout_df.to_csv(path_to_store_holdout_scores, index=False)
        logging.info("The scores for the holdout set were saved at {}".format(path_to_store_holdout_scores))
