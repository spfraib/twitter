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
    parser.add_argument("--parquet_folder_path", type=str, help="Path to the training folder containing parquet files.")
    parser.add_argument("--eval_data_path", type=str, help="Path to the evaluation data. Must be in csv format.")
    parser.add_argument("--preprocessed_input", type=int, default=0,
                        help="Whether the text input was preprocessed")
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--run_name", type=str,
                        help="Define customized run name to find it back in saved models folder.")
    parser.add_argument("--num_train_epochs", type=int)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--best_model_path", type=str)
    parser.add_argument("--label_to_train_on", type=str)
    parser.add_argument("--output_dir", type=str, help="Define a folder to store the saved models",
                        default=OUTPUTS_FOLDER)
    args = parser.parse_args()
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

def convert_labels_to_int(label):
    if label == 'yes':
        return 1
    elif label == 'no':
        return 0

if __name__ == "__main__":
    # Define args from command line
    args = get_args_from_command_line()
    # import new labels
    new_labels_df = pd.read_parquet(os.path.join(args.parquet_folder_path, 'training_data1.parquet'))
    new_labels_df.columns = ['tweet_id', 'text', 'sel_sample', 'sel_class', 'sel_threshold', 'model_score', 'model', 'Unnamed: 0', 'is_unemployed', 'lost_job_1mo', 'job_search', 'is_hired_1mo', 'job_offer']
    # import eval data
    eval_df = pd.read_csv(args.eval_data_path)
    eval_df = eval_df[['text','class']]
    eval_df.columns = ['text', 'labels']
    # format the data for finetuning
    new_labels_df = new_labels_df.loc[new_labels_df['sel_class'] == args.label_to_train_on].reset_index(drop=True)
    new_labels_df = new_labels_df[['text', args.label_to_train_on]]
    new_labels_df.columns = ['text', 'labels']
    new_labels_df['labels'] = new_labels_df['labels'].apply(convert_labels_to_int)
    new_labels_df = new_labels_df[new_labels_df['labels'].notna()].reset_index(drop=True)
    print(new_labels_df.head())
    # import model
    model = ClassificationModel(args.model_type, args.best_model_path,
                                args={'evaluate_during_training': True, 'evaluate_during_training_verbose': True,
                                      'num_train_epochs': 20})
    eval_metrics = {
        "precision": sklearn.metrics.precision_score,
        "recall": sklearn.metrics.recall_score,
        "f1": sklearn.metrics.f1_score}

    path_to_store_model = prepare_filepath_for_storing_model(output_dir=args.output_dir, run_name=args.run_name)
    model.train_model(new_labels_df, eval_df=eval_df, output_dir=path_to_store_model, **eval_metrics)