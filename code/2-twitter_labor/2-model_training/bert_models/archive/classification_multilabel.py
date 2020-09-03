import os
import logging
import argparse
from simpletransformers.classification import ClassificationModel, MultiLabelClassificationModel
import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_list_like
from sklearn.model_selection import train_test_split
import sklearn
import numpy as np
from sklearn import metrics

OUTPUTS_FOLDER = os.path.join(os.path.dirname(__file__), "outputs/")

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()

    # necessary
    parser.add_argument("--data_path", type=str, help="Path to the training data. Must be in pickle format.")
    parser.add_argument("--preprocessed_input", type=int, default=0,
                        help="Whether the text input was preprocessed")
    parser.add_argument("--num_labels", type=int, default=5)
    parser.add_argument("--run_name", type=str,
                        help="Define customized run name to find it back in saved models folder.")
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
    elif not is_list_like(df['labels']):
        raise ValueError(
            'The "labels" column should be of list type')


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


def convert_labels_to_int(label):
    if label == 'yes':
        return 1
    elif label == 'no':
        return 0


def reshape_output(output_array, label, threshold):
    if label == 'lost_job_1mo':
        output_array = output_array[:, 0]
    elif label == 'is_unemployed':
        output_array = output_array[:, 1]
    elif label == 'job_search':
        output_array = output_array[:, 2]
    elif label == 'is_hired_1mo':
        output_array = output_array[:, 3]
    elif label == 'job_offer':
        output_array = output_array[:, 4]
    label_pred_array = np.array([1 if output_array[i] > threshold else 0 for i in range(output_array.shape[0])])
    return [output_array, label_pred_array]


def precision_lost_job_1mo(true_labels, model_outputs):
    true_labels_lost_job_1mo = true_labels[:, 0]
    model_outputs_lost_job_1mo = reshape_output(model_outputs, 'lost_job_1mo', 0.5)[1]
    precision_lost_job_1mo = metrics.precision_score(true_labels_lost_job_1mo, model_outputs_lost_job_1mo)
    return precision_lost_job_1mo


def precision_is_unemployed(true_labels, model_outputs):
    true_labels_is_unemployed = true_labels[:, 1]
    model_outputs_is_unemployed = reshape_output(model_outputs, 'is_unemployed', 0.5)[1]
    precision_is_unemployed = metrics.precision_score(true_labels_is_unemployed, model_outputs_is_unemployed)
    return precision_is_unemployed


def precision_job_search(true_labels, model_outputs):
    true_labels_job_search = true_labels[:, 2]
    model_outputs_job_search = reshape_output(model_outputs, 'job_search', 0.5)[1]
    precision_job_search = metrics.precision_score(true_labels_job_search, model_outputs_job_search)
    return precision_job_search


def precision_is_hired_1mo(true_labels, model_outputs):
    true_labels_is_hired_1mo = true_labels[:, 3]
    model_outputs_is_hired_1mo = reshape_output(model_outputs, 'is_hired_1mo', 0.5)[1]
    precision_is_hired_1mo = metrics.precision_score(true_labels_is_hired_1mo, model_outputs_is_hired_1mo)
    return precision_is_hired_1mo


def precision_job_offer(true_labels, model_outputs):
    true_labels_job_offer = true_labels[:, 4]
    model_outputs_job_offer = reshape_output(model_outputs, 'job_offer', 0.5)[1]
    precision_job_offer = metrics.precision_score(true_labels_job_offer, model_outputs_job_offer)
    return precision_job_offer


def recall_lost_job_1mo(true_labels, model_outputs):
    true_labels_lost_job_1mo = true_labels[:, 0]
    model_outputs_lost_job_1mo = reshape_output(model_outputs, 'lost_job_1mo', 0.5)[1]
    recall_lost_job_1mo = metrics.recall_score(true_labels_lost_job_1mo, model_outputs_lost_job_1mo)
    return recall_lost_job_1mo


def recall_is_unemployed(true_labels, model_outputs):
    true_labels_is_unemployed = true_labels[:, 1]
    model_outputs_is_unemployed = reshape_output(model_outputs, 'is_unemployed', 0.5)[1]
    recall_is_unemployed = metrics.recall_score(true_labels_is_unemployed, model_outputs_is_unemployed)
    return recall_is_unemployed


def recall_job_search(true_labels, model_outputs):
    true_labels_job_search = true_labels[:, 2]
    model_outputs_job_search = reshape_output(model_outputs, 'job_search', 0.5)[1]
    recall_job_search = metrics.recall_score(true_labels_job_search, model_outputs_job_search)
    return recall_job_search


def recall_is_hired_1mo(true_labels, model_outputs):
    true_labels_is_hired_1mo = true_labels[:, 3]
    model_outputs_is_hired_1mo = reshape_output(model_outputs, 'is_hired_1mo', 0.5)[1]
    recall_is_hired_1mo = metrics.recall_score(true_labels_is_hired_1mo, model_outputs_is_hired_1mo)
    return recall_is_hired_1mo


def recall_job_offer(true_labels, model_outputs):
    true_labels_job_offer = true_labels[:, 4]
    model_outputs_job_offer = reshape_output(model_outputs, 'job_offer', 0.5)[1]
    recall_job_offer = metrics.recall_score(true_labels_job_offer, model_outputs_job_offer)
    return recall_job_offer


def auc_lost_job_1mo(true_labels, model_outputs):
    true_labels_lost_job_1mo = true_labels[:, 0]
    model_outputs_lost_job_1mo = reshape_output(model_outputs, 'lost_job_1mo', 0.5)[0]
    fpr, tpr, thresholds = metrics.roc_curve(true_labels_lost_job_1mo, model_outputs_lost_job_1mo)
    auc_lost_job_1mo = metrics.auc(fpr, tpr)
    return auc_lost_job_1mo


def auc_is_unemployed(true_labels, model_outputs):
    true_labels_is_unemployed = true_labels[:, 1]
    model_outputs_is_unemployed = reshape_output(model_outputs, 'is_unemployed', 0.5)[1]
    fpr, tpr, thresholds = metrics.roc_curve(true_labels_is_unemployed, model_outputs_is_unemployed)
    auc_is_unemployed = metrics.auc(fpr, tpr)
    return auc_is_unemployed


def auc_job_search(true_labels, model_outputs):
    true_labels_job_search = true_labels[:, 2]
    model_outputs_job_search = reshape_output(model_outputs, 'job_search', 0.5)[1]
    fpr, tpr, thresholds = metrics.roc_curve(true_labels_job_search, model_outputs_job_search)
    auc_job_search = metrics.auc(fpr, tpr)
    return auc_job_search


def auc_is_hired_1mo(true_labels, model_outputs):
    true_labels_is_hired_1mo = true_labels[:, 3]
    model_outputs_is_hired_1mo = reshape_output(model_outputs, 'is_hired_1mo', 0.5)[1]
    fpr, tpr, thresholds = metrics.roc_curve(true_labels_is_hired_1mo, model_outputs_is_hired_1mo)
    auc_is_hired_1mo = metrics.auc(fpr, tpr)
    return auc_is_hired_1mo


def auc_job_offer(true_labels, model_outputs):
    true_labels_job_offer = true_labels[:, 4]
    model_outputs_job_offer = reshape_output(model_outputs, 'job_offer', 0.5)[1]
    fpr, tpr, thresholds = metrics.roc_curve(true_labels_job_offer, model_outputs_job_offer)
    auc_job_offer = metrics.auc(fpr, tpr)
    return auc_job_offer


if __name__ == "__main__":
    # Define args from command line
    args = get_args_from_command_line()
    args = check_command_line_args(args)

    # Display the arguments
    logger.info(args)

    # Import the data
    all_data_df = pd.read_pickle(args.data_path)
    all_data_df.columns = ['tweet_id', 'text', 'is_unemployed', 'lost_job_1mo', 'job_search', 'is_hired_1mo',
                           'job_offer']
    all_data_df = all_data_df[
        ['tweet_id', 'text', 'lost_job_1mo', 'is_unemployed', 'job_search', 'is_hired_1mo', 'job_offer']]
    # convert labels to int
    for label in ['is_unemployed', 'lost_job_1mo', 'job_search', 'is_hired_1mo', 'job_offer']:
        all_data_df[label] = all_data_df[label].apply(convert_labels_to_int)
        all_data_df = all_data_df[all_data_df[label].notna()].reset_index(drop=True)
    # formatting
    all_data_df['labels'] = all_data_df[
        ['lost_job_1mo', 'is_unemployed', 'job_search', 'is_hired_1mo', 'job_offer']].values.tolist()

    # train/test split
    train_df = all_data_df.sample(frac=0.7, random_state=0)
    eval_df = all_data_df.drop(train_df.index)
    train_df = train_df.reset_index(drop=True)
    eval_df = eval_df.reset_index(drop=True)
    train_df = train_df[['text', 'labels']]
    eval_df = eval_df[['text', 'labels']]
    print("********** Train shape: ", train_df.shape[0], " **********")
    print("********** Eval shape: ", eval_df.shape[0], " **********")
    # Make sure the DataFrame contains the necessary columns
    verify_data_format(train_df)
    verify_data_format(eval_df)
    # Make sure the columns have the right type
    verify_column_type(train_df)
    verify_column_type(eval_df)
    # Create a ClassificationModel
    model = MultiLabelClassificationModel(args.model_type, args.model_name, num_labels=args.num_labels,
                                args={'evaluate_during_training': True, 'evaluate_during_training_verbose': True,
                                      'num_train_epochs': args.num_train_epochs, "use_early_stopping": True,
                                      "early_stopping_patience": 3,
                                      "early_stopping_delta": 0, "early_stopping_metric": "eval_loss",
                                      "early_stopping_metric_minimize": True})
    # Define evaluation metrics
    eval_metrics = {
        "precision_lost_job_1mo": precision_lost_job_1mo,
        "precision_is_unemployed": precision_is_unemployed,
        "precision_job_search": precision_job_search,
        "precision_is_hired_1mo": precision_is_hired_1mo,
        "precision_job_offer": precision_job_offer,
        "recall_lost_job_1mo": recall_lost_job_1mo,
        "recall_is_unemployed": recall_is_unemployed,
        "recall_job_search": recall_job_search,
        "recall_is_hired_1mo": recall_is_hired_1mo,
        "recall_job_offer": recall_job_offer,
        "auc_lost_job_1mo": auc_lost_job_1mo,
        "auc_is_unemployed": auc_is_unemployed,
        "auc_job_search": auc_job_search,
        "auc_is_hired_1mo": auc_is_hired_1mo,
        "auc_job_offer": auc_job_offer
    }
    # Train the model
    path_to_store_model = prepare_filepath_for_storing_model(output_dir=args.output_dir, run_name=args.run_name)
    model.train_model(train_df, eval_df=eval_df, output_dir=path_to_store_model, **eval_metrics)
    logging.info("The training of the model is done")
