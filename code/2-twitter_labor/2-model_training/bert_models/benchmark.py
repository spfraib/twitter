import pandas as pd
import argparse
import os
from collections import defaultdict
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str,
                        default="US")
    parser.add_argument("--results_folder", type=str,
                        default="US")
    args = parser.parse_args()
    return args


def build_average_auc_dict(results_dict: dict, model_type: str) -> dict:
    """
    Compute the average AUC value for each label and across models.
    Adapted from https://stackoverflow.com/questions/36053321/python-average-all-key-values-with-same-key-to-dictionary
    """
    by_label = defaultdict(list)
    for model, mrs in results_dict[model_type].items():
        for label_key, auc_value in mrs.items():
            by_label[label_key].append(auc_value)
    return {label_key: sum(auc_value_list) / len(auc_value_list) for label_key, auc_value_list in by_label.items()}


def compare_model_results(results_dict_1: dict, results_dict_2: dict, model_types_list: list):
    """ Compare average AUC results and print best results for each label."""
    for label in ['lost_job_1mo', 'is_hired_1mo', 'is_unemployed', 'job_offer', 'job_search']:
        label_key = f'auc_{label}'
        diff = abs((results_dict_1[label_key] - results_dict_2[label_key]) / results_dict_1[label_key])*100
        logger.info('\n')
        if results_dict_1[label_key] > results_dict_2[label_key]:
            logger.info(f'Model {model_types_list[0]} better for label {label} by {diff}%')
        else:
            logger.info(f'Model {model_types_list[1]} better for label {label} by {diff}%')
        logger.info(f'AUC for label {label} from model {model_types_list[0]}: {results_dict_1[label_key]}')
        logger.info(f'AUC for label {label} from model {model_types_list[1]}: {results_dict_2[label_key]}')


if __name__ == '__main__':
    args = get_args_from_command_line()
    results_folders_list = os.listdir(args.results_folder)
    print(results_folders_list)
    job_ids_list = [x.split('_')[1] for x in results_folders_list]
    print(job_ids_list)
    # keep 10 latest jobs
    indexes_to_keep_list = sorted(range(len(job_ids_list)), key=lambda x: job_ids_list[x])[-10:]
    print(indexes_to_keep_list)
    results_dict = dict()
    model_types_list = list()
    for index in indexes_to_keep_list:
        result_folder_name_str = results_folders_list[index]
        model_type_str = f"{result_folder_name_str.split('_')[0]}_{result_folder_name_str.split('_')[2]}"
        if model_type_str not in model_types_list:
            model_types_list.append(model_type_str)
        if model_type_str not in results_dict.keys():
            results_dict[model_type_str] = dict()
        results_dict[model_type_str][result_folder_name_str] = dict()
        for label in ['lost_job_1mo', 'is_hired_1mo', 'is_unemployed', 'job_offer', 'job_search']:
            df = pd.read_csv(os.path.join(args.results_folder, result_folder_name_str, f'val_{label}_evaluation.csv'),
                             index_col=0)
            results_dict[model_type_str][result_folder_name_str][f'auc_{label}'] = float(df['value']['auc'])
    # get average AUC for each label and model type
    if len(model_types_list) != 2:
        logger.error("More than two models are being compared.")
    average_auc_model_1_dict = build_average_auc_dict(results_dict=results_dict, model_type=model_types_list[0])
    average_auc_model_2_dict = build_average_auc_dict(results_dict=results_dict, model_type=model_types_list[1])
    compare_model_results(results_dict_1=average_auc_model_1_dict, results_dict_2=average_auc_model_2_dict,
                          model_types_list=model_types_list)
