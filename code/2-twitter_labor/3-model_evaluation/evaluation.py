import pandas as pd
import argparse
import os
from collections import defaultdict
import logging
import statistics
import socket
from pathlib import Path
import re

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str,
                        default="US")
    parser.add_argument("--data_folder", type=str,
                        default="jan5_iter0")
    # parser.add_argument("--run_name", type=str, help="Name of the output CSV containing results", default='test')


    args = parser.parse_args()
    return args


def build_auc_dict(results_dict: dict, model_type: str) -> dict:
    """
    Compute AUC statistics for each label and across models.
    Adapted from https://stackoverflow.com/questions/36053321/python-average-all-key-values-with-same-key-to-dictionary
    """
    by_label = defaultdict(list)
    for model, mrs in results_dict[model_type].items():
        for label_key, auc_value in mrs.items():
            by_label[label_key].append(auc_value)

    return {label_key: {'mean': sum(auc_value_list) / len(auc_value_list),
                        #'std': statistics.stdev(auc_value_list),
                        'min': min(auc_value_list),
                        'max': max(auc_value_list)
                        } for label_key, auc_value_list in by_label.items()}


# def output_results(results_dict: dict):
#     """Log AUC statistics for each label"""
#     for label in ['lost_job_1mo', 'is_hired_1mo', 'is_unemployed', 'job_offer', 'job_search']:
#         label_key = f'auc_{label}'
#         logger.info(f"******** Label: {label} ********")
#         logger.info(f"Mean AUC: {results_dict[label_key]['mean']}")
#         logger.info(f"Standard deviation: {results_dict[label_key]['std']}")
#         logger.info(f"Minimum: {results_dict[label_key]['min']}")
#         logger.info(f"Maximum: {results_dict[label_key]['max']}")
#         logger.info('\n')

# def compare_model_results(results_dict_1: dict, results_dict_2: dict, model_types_list: list):
#     """ Compare average AUC results and print best results for each label."""
#     for label in ['lost_job_1mo', 'is_hired_1mo', 'is_unemployed', 'job_offer', 'job_search']:
#         label_key = f'auc_{label}'
#         diff = abs((results_dict_1[label_key] - results_dict_2[label_key]) / results_dict_1[label_key])*100
#         logger.info('\n')
#         if results_dict_1[label_key] > results_dict_2[label_key]:
#             logger.info(f'Model {model_types_list[0]} better for label {label} by {diff}%')
#         else:
#             logger.info(f'Model {model_types_list[1]} better for label {label} by {diff}%')
#         logger.info(f'AUC for label {label} from model {model_types_list[0]}: {results_dict_1[label_key]}')
#         logger.info(f'AUC for label {label} from model {model_types_list[1]}: {results_dict_2[label_key]}')


if __name__ == '__main__':
    args = get_args_from_command_line()
    if 'manuto' in socket.gethostname().lower():
        output_path = f'/home/manuto/Documents/world_bank/bert_twitter_labor/twitter-labor-data/data/train_test/{args.country_code}/{args.data_folder}/evaluation'
        results_folder = f'/home/manuto/Documents/world_bank/bert_twitter_labor/twitter-labor-data/data/train_test/{args.country_code}/{args.data_folder}/train_test/results'
    else:
        output_path = f'/scratch/mt4493/twitter_labor/twitter-labor-data/data/{args.data_folder}/{args.country_code}/evaluation'
        results_folder = f'/scratch/mt4493/twitter_labor/twitter-labor-data/data/{args.data_folder}/{args.country_code}/train_test/results'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    results_folders_list = os.listdir(results_folder)
    results_dict = dict()

    for model in ['neuralmind-bert-base-portuguese-cased', 'DeepPavlov-bert-base-cased-conversational', 'dccuchile-bert-base-spanish-wwm-cased']:
        for seed in range(1,16):
            r = re.compile(f'{model}_[0-9]+_seed-{str(seed)}$')
            folder_name_str = list(filter(r.match, results_folders_list))
            if len(folder_name_str) > 0:
                folder_name_str = list(filter(r.match, results_folders_list))[0]
                for label in ['lost_job_1mo', 'is_hired_1mo', 'is_unemployed', 'job_offer', 'job_search']:
                    if label not in results_dict.keys():
                        results_dict[label] = dict()
                    path_data = os.path.join(results_folder, folder_name_str, f'val_{label}_evaluation.csv')
                    if Path(path_data).exists():
                        df = pd.read_csv(
                            os.path.join(results_folder, folder_name_str, f'val_{label}_evaluation.csv'),
                            index_col=0)
                        results_dict[label][f'auc_{model}_{str(seed)}'] = float(df['value']['auc'])
    results_df = pd.DataFrame.from_dict(results_dict)
    results_df = results_df.round(3)
    results_df = results_df.reset_index()
    results_df['model'] = results_df['index'].apply(lambda x: x.split('_')[1])
    results_df['seed'] = results_df['index'].apply(lambda x: x.split('_')[2])
    results_df = results_df[['model','seed', 'lost_job_1mo', 'is_hired_1mo', 'is_unemployed', 'job_offer', 'job_search']]
    results_df.to_csv(os.path.join(output_path, 'auc_results.csv'))
    results_df = results_df.set_index(['seed'])
    results_df = results_df[['lost_job_1mo', 'is_hired_1mo', 'is_unemployed', 'job_offer', 'job_search']]
    print(results_df.idxmax())
    print(results_df.max())
    #logger.info(results_dict)


    # logger.info(results_folders_list)
    # job_ids_list = [x.split('_')[1] for x in results_folders_list]
    # logger.info(job_ids_list)
    # # keep 10 latest jobs
    # indexes_to_keep_list = sorted(range(len(job_ids_list)), key=lambda x: job_ids_list[x])[-15:]
    # logger.info(indexes_to_keep_list)
    # results_dict = dict()
    # model_types_list = list()
    # for index in indexes_to_keep_list:
    #     result_folder_name_str = results_folders_list[index]
    #     print(result_folder_name_str)
    #     model_type_str = f"{result_folder_name_str.split('_')[0]}_{result_folder_name_str.split('_')[2]}"
    #     if model_type_str not in model_types_list:
    #         model_types_list.append(model_type_str)
    #     if model_type_str not in results_dict.keys():
    #         results_dict[model_type_str] = dict()
    #     # results_dict[model_type_str] = dict()
    #     for label in ['lost_job_1mo', 'is_hired_1mo', 'is_unemployed', 'job_offer', 'job_search']:
    #         path_data = os.path.join(args.results_folder, result_folder_name_str, f'val_{label}_evaluation.csv')
    #         if Path(path_data).exists():
    #             df = pd.read_csv(os.path.join(args.results_folder, result_folder_name_str, f'holdout_{label}_evaluation.csv'),
    #                              index_col=0)
    #             results_dict[model_type_str][f'auc_{label}'] = float(df['value']['auc'])
    # # get average AUC for each label and model type
    # # if len(model_types_list) != 2:
    # #     logger.error("More than two models are being compared.")
    # logger.info(model_types_list)
    # logger.info(results_dict)
    # #average_auc_model_1_dict = build_auc_dict(results_dict=results_dict, model_type=model_types_list[0])
    # results_df = pd.DataFrame.from_dict(results_dict, orient='index')
    # results_df = results_df.round(3)
    # print(results_df.head(n=15))
    # # if not os.path.exists(output_path):
    # #     os.makedirs(output_path)
    # # results_df.to_csv(os.path.join(output_path, f'{args.run_name}.csv'))

