"""
This script launches the training of a Transformer-based text classification model and saves the trained model
in a designated folder.

The train and evaluation input data needs to be in CSV format and contain at least two columns (named 'text' and 'class').

By default, we use BERT base and Conversational BERT.
The default model specs are listed here: https://github.com/ThilinaRajapakse/simpletransformers#args-explained


How to use the script in the command line:
python3 classification.py
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
python classification.py \
--train_data_path twitter-labor-data/data/jun_27_iter1/BERT/new_train_test_split_with_data_iter0/train_is_hired_1mo.csv \
--eval_data_path twitter-labor-data/data/jun_27_iter1/BERT/new_train_test_split_with_data_iter0/val_is_hired_1mo.csv \
--output_dir trained_bert-base-cased_is_hired_1mo_jun_27_iter1 \
--num_labels 2 \
--num_train_epochs 20
--model_name bert \
--model_type bert-base-cased
"""
import os
import logging
import argparse
from simpletransformers.classification import ClassificationModel
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
import sklearn

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
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int)
    parser.add_argument("--model_name", type=str,
                        help="Select the model to use.", default='bert')
    parser.add_argument("--model_type", type=str, default='bert-base-cased')
    parser.add_argument("--output_dir", type=str, help="Define a folder to store the saved models",
                        default=OUTPUTS_FOLDER)

    args = parser.parse_args()
    return args


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


if __name__ == "__main__":
    # Define args from command line
    args = get_args_from_command_line()
    args = check_command_line_args(args)

    # Display the arguments
    logger.info(args)
    # Import data
    train_df = pd.read_csv(args.train_data_path, lineterminator='\n')
    eval_df = pd.read_csv(args.eval_data_path, lineterminator='\n')
    # Reformat the data
    train_df = train_df[["text", "class"]]
    eval_df = eval_df[["text", "class"]]
    train_df.columns = ['text', 'labels']
    eval_df.columns = ['text', 'labels']

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
    # Create a ClassificationModel
    model = ClassificationModel(args.model_type, args.model_name, num_labels=args.num_labels,
                                args={'overwrite_output_dir': True, 'evaluate_during_training': True,
                                      'save_model_every_epoch': False, 'save_eval_checkpoints': False,
                                      'output_dir': path_to_store_model, 'best_model_dir': path_to_store_best_model,
                                      'evaluate_during_training_verbose': True,
                                      'num_train_epochs': args.num_train_epochs, "use_early_stopping": True,
                                      "early_stopping_patience": 3,
                                      "early_stopping_delta": 0, "early_stopping_metric": "eval_loss",
                                      "early_stopping_metric_minimize": True})
    # Define evaluation metrics
    eval_metrics = {
        "precision": sklearn.metrics.precision_score,
        "recall": sklearn.metrics.recall_score,
        "f1": sklearn.metrics.f1_score
    }
    # Train the model
    model.train_model(train_df, eval_df=eval_df, output_dir=path_to_store_model, **eval_metrics)
    logging.info("The training of the model is done")