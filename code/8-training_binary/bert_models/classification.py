"""
This script launches the training of a Transformer-based text classification model and saves the trained model
in a designated folder.

The train and evaluation data needs to be in a Pandas DataFrame containing at least two columns.
The DataFrame should have a header and contain a 'text' and a 'labels' column.

By default, we use BERT base.
The default model specs are listed here: https://github.com/ThilinaRajapakse/simpletransformers#args-explained


How to use the script in the command line:
python3 classification.py
    --data_path <DATA_PATH> \
    --num_labels <NUM_LABELS> \
    --run_name <RUN_NAME> \
    --model_name <MODEL_NAME> \
    --model_type <MODEL_TYPE> \
    --output_dir <OUTPUT_DIR> \
    --test_size <TEST_SIZE> \
    --random_state <RANDOM_STATE> \
    --shuffle <SHUFFLE>
Where:
<DATA_PATH>: Path to the CSV used as input file (compulsory)

<NUM_LABELS>: Number of classes in the classification problem (compulsory)

<RUN_NAME>: Customized name to differentiate different saved models (compulsory)

<MODEL_NAME>: Name of the model to use

<MODEL_TYPE>: Type of model to be used (compulsory)
A list can be found here: https://github.com/ThilinaRajapakse/simpletransformers#current-pretrained-models)

<OUTPUT_DIR>: Path to the folder where the trained models and the data after train_test_split will be stored.

<TEST_SIZE>: Test size in the train-test split (optional)

<RANDOM_STATE>: Random state for the train test split (optional)

<SHUFFLE>: Shuffle for the train test split (optional)

<TEST>: Option to directly get the model performance from the test set (optional)


Example usage:
python classification.py \
 --data_path /Users/manueltonneau/Documents/Humboldt/Second_semester/Master_thesis/data/labeled_data/FinancialPhraseBank-v1.0/test.csv \
 --output_dir /Users/manueltonneau/Documents/creatext_code/creatext_sandbox/sandbox_projects/Transformers_for_text_classification/ \
 --num_labels 3 \
 --run_name some_funny_name \
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

OUTPUTS_FOLDER = os.path.join(os.path.dirname(__file__), "outputs/")

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()

    # necessary
    parser.add_argument("--train_data_path", type=str, help="Path to the training data. Must be in csv format.", default="")
    parser.add_argument("--eval_data_path", type=str, help="Path to the evaluation data. Must be in csv format.", default="")
    parser.add_argument("--train_all_labels", type=int, default=0)
    parser.add_argument("--all_labels_path", type=str, default="")
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--preprocessed_input", type=int, default=0,
                        help="Whether the text input was preprocessed")
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--run_name", type=str,
                        help="Define customized run name to find it back in saved models folder.")
    parser.add_argument("--num_train_epochs", type=int)
    parser.add_argument("--model_name", type=str,
                        help="Select the model to use.", default='bert')
    parser.add_argument("--model_type", type=str, default='bert-base-cased')
    parser.add_argument("--output_dir", type=str, help="Define a folder to store the saved models",
                        default=OUTPUTS_FOLDER)
    parser.add_argument("--add_new_labels", type=bool, default=False)
    parser.add_argument("--parquet_folder_path", type=str, default="")
    parser.add_argument("--label_to_train_on", type=str, default="")
    parser.add_argument("--new_train_test_split", type=int, default=0)


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


def prepare_filepath_for_storing_model(output_dir: str, run_name: str) -> str:
    """Prepare the filepath where the trained model will be stored.

    :param output_dir: Directory where to store outputs (dataset splits + trained models).
    :param run_name: Customized name to differentiate different saved models.
    :return: path_to_store_model: Path where to store the trained model.
    """
    path_to_store_model = os.path.join(output_dir, run_name, 'models')
    if not os.path.exists(path_to_store_model):
        os.makedirs(path_to_store_model)
    return path_to_store_model

def prepare_filepath_for_storing_best_model(path_to_store_model: str) -> str:
    path_to_store_best_model = os.path.join(path_to_store_model, 'best_model')
    if not os.path.exists(path_to_store_best_model):
        os.makedirs(path_to_store_best_model)
    return path_to_store_best_model

def convert_labels_to_int(label):
    if label == 'yes':
        return 1
    elif label == 'no':
        return 0

if __name__ == "__main__":
    # Define args from command line
    args = get_args_from_command_line()
    args = check_command_line_args(args)

    # Display the arguments
    logger.info(args)

    # Import the data
    # if args.train_all_labels == 1:
    #     # Import the data
    #     all_data_df = pd.read_pickle(args.all_labels_path)
    #     all_data_df.columns = ['tweet_id', 'text', 'is_unemployed', 'lost_job_1mo', 'job_search', 'is_hired_1mo',
    #                            'job_offer']
    #     all_data_df = all_data_df[
    #         ['tweet_id', 'text', 'lost_job_1mo', 'is_unemployed', 'job_search', 'is_hired_1mo', 'job_offer']]
    #     # convert labels to int
    #     for label in ['is_unemployed', 'lost_job_1mo', 'job_search', 'is_hired_1mo', 'job_offer']:
    #         all_data_df[label] = all_data_df[label].apply(convert_labels_to_int)
    #         all_data_df = all_data_df[all_data_df[label].notna()].reset_index(drop=True)
    #     # train/test split
    #     train_df = all_data_df.sample(frac=0.7, random_state=0)
    #     eval_df = all_data_df.drop(train_df.index)
    #     train_df = train_df.reset_index(drop=True)
    #     eval_df = eval_df.reset_index(drop=True)
    #     train_df = train_df[['text', args.label]]
    #     eval_df = eval_df[['text', args.label]]
    #     train_df.columns = ['text','labels']
    #     eval_df.columns = ['text', 'labels']
    #     print("********** Train shape: ", train_df.shape[0], " **********")
    #     print("********** Eval shape: ", eval_df.shape[0], " **********")
    # else:
    train_df = pd.read_csv(args.train_data_path, lineterminator='\n')
    eval_df = pd.read_csv(args.eval_data_path, lineterminator='\n')
    # Reformat the data
    if args.preprocessed_input == 1:
        train_df = train_df[["ProcessedText", "class"]]
        eval_df = eval_df[["ProcessedText", "class"]]
        args.run_name = args.run_name + "_preprocessed"
    else:
        train_df = train_df[["text", "class"]]
        eval_df = eval_df[["text", "class"]]
    train_df.columns = ['text', 'labels']
    eval_df.columns = ['text', 'labels']
    print("********** Initial train shape: ", train_df.shape[0], " **********")
    print("********** Initial eval shape: ", eval_df.shape[0], " **********")
        # if args.add_new_labels:
        #     if args.new_train_test_split == 1:
        #         new_labels_df = pd.read_parquet(os.path.join(args.parquet_folder_path, 'training_data1.parquet'))
        #         new_labels_df.columns = ['tweet_id', 'text', 'sel_sample', 'sel_class', 'sel_threshold', 'model_score',
        #                              'model',
        #                              'Unnamed: 0', 'is_unemployed', 'lost_job_1mo', 'job_search', 'is_hired_1mo',
        #                              'job_offer']
        #         new_labels_df = new_labels_df[['text', args.label_to_train_on]]
        #         new_labels_df.columns = ['text', 'labels']
        #         new_labels_df['labels'] = new_labels_df['labels'].apply(convert_labels_to_int)
        #         new_labels_df = new_labels_df[new_labels_df['labels'].notna()].reset_index(drop=True)
        #         all_data_df = pd.concat([train_df, eval_df, new_labels_df]).reset_index(drop=True)
        #         train_df = all_data_df.sample(frac=0.7, random_state=0)
        #         eval_df = all_data_df.drop(train_df.index)
        #         train_df = train_df.reset_index(drop=True)
        #         eval_df = eval_df.reset_index(drop=True)
        #         print("********** Combined train shape: ", train_df.shape[0], " **********")
        #         print("********** Combined eval shape: ", eval_df.shape[0], " **********")
        #     else:
        #         new_labels_df = pd.read_parquet(os.path.join(args.parquet_folder_path, 'training_data1.parquet'))
        #         new_labels_df.columns = ['tweet_id', 'text', 'sel_sample', 'sel_class', 'sel_threshold', 'model_score', 'model',
        #                          'Unnamed: 0', 'is_unemployed', 'lost_job_1mo', 'job_search', 'is_hired_1mo',
        #                          'job_offer']
        #         new_labels_df = new_labels_df[['text', args.label_to_train_on]]
        #         new_labels_df.columns = ['text', 'labels']
        #         new_labels_df['labels'] = new_labels_df['labels'].apply(convert_labels_to_int)
        #         new_labels_df = new_labels_df[new_labels_df['labels'].notna()].reset_index(drop=True)
        #         train_df = pd.concat([train_df, new_labels_df]).reset_index(drop=True)
        #         print("********** Combined data shape: ", train_df.shape[0], " **********")
    # Make sure the DataFrame contains the necessary columns
    verify_data_format(train_df)
    verify_data_format(eval_df)
    # Make sure the columns have the right type
    verify_column_type(train_df)
    verify_column_type(eval_df)

    path_to_store_model = prepare_filepath_for_storing_model(output_dir=args.output_dir, run_name=args.run_name)
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
#         "roc_auc": sklearn.metrics.roc_auc_score, #give an error: ValueError: x is neither increasing nor decreasing
#         "auc": sklearn.metrics.auc
    }
    # Train the model
    model.train_model(train_df, eval_df=eval_df, output_dir=path_to_store_model, **eval_metrics)
    logging.info("The training of the model is done")
