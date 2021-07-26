import os
import torch
import onnxruntime as ort
import pandas as pd
import numpy as np
import os
import time
import torch.nn.functional as F
import onnx
import getpass
from transformers import AutoTokenizer
import time
import pyarrow.parquet as pq
from glob import glob
import os
import numpy as np
import argparse
import logging
import socket
import multiprocessing
from functools import reduce

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
print('libs loaded')

parser = argparse.ArgumentParser()

parser.add_argument("--input_path", type=str, help="path to input data")
parser.add_argument("--output_path", type=str, help="path where inference csv is saved")
parser.add_argument("--country_code", type=str, help="path where inference csv is saved")
parser.add_argument("--iteration_number", type=int)
parser.add_argument("--method", type=int)
parser.add_argument("--debug_mode", type=bool, help="fast debug mode", default=True)
parser.add_argument("--drop_duplicates", type=bool, help="drop duplicated tweets from parquet files", default=False)
parser.add_argument("--resume", type=int, help="resuming a run, 0 or 1")


args = parser.parse_args()

print(args)

DEBUG_MODE = args.debug_mode

global_start = time.time()

####################################################################################################################################
# HELPER FUNCTIONS
####################################################################################################################################

# inference
def get_tokens(tokens_dict, i):
    i_tokens_dict = dict()
    for key in ['input_ids', 'token_type_ids', 'attention_mask']:
        i_tokens_dict[key] = tokens_dict[key][i]
    tokens = {name: np.atleast_2d(value) for name, value in i_tokens_dict.items()}
    return tokens


def inference(onnx_model, model_dir, examples):
    quantized_str = ''
    if 'quantized' in onnx_model:
        quantized_str = 'quantized'
    onnx_inference = []
    #     pytorch_inference = []
    # onnx session
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 1
    # options.inter_op_num_threads = multiprocessing.cpu_count()

    print(onnx_model)
    ort_session = ort.InferenceSession(onnx_model, options)

    # pytorch pretrained model and tokenizer
    if 'bertweet' in onnx_model:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, normalization=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

    tokenizer_str = "TokenizerFast"

    print("**************** {} ONNX inference with batch tokenization and with {} tokenizer****************".format(
        quantized_str, tokenizer_str))
    start_onnx_inference_batch = time.time()
    start_batch_tokenization = time.time()
    tokens_dict = tokenizer.batch_encode_plus(examples, max_length=128)
    total_batch_tokenization_time = time.time() - start_batch_tokenization
    total_inference_time = 0
    total_build_label_time = 0
    for i in range(len(examples)):
        """
        Onnx inference with batch tokenization
        """

        if i % 100 == 0:
            print(f'[inference: {i} out of {str(len(examples))}')

        tokens = get_tokens(tokens_dict, i)
        # inference
        start_inference = time.time()
        ort_outs = ort_session.run(None, tokens)
        total_inference_time = total_inference_time + (time.time() - start_inference)
        # build label
        start_build_label = time.time()
        torch_onnx_output = torch.tensor(ort_outs[0], dtype=torch.float32)
        onnx_logits = F.softmax(torch_onnx_output, dim=1)
        logits_label = torch.argmax(onnx_logits, dim=1)
        label = logits_label.detach().cpu().numpy()
        #         onnx_inference.append(label[0])
        onnx_inference.append(onnx_logits.detach().cpu().numpy()[0].tolist())
        total_build_label_time = total_build_label_time + (time.time() - start_build_label)
    #         print(i, label[0], onnx_logits.detach().cpu().numpy()[0].tolist(), type(onnx_logits.detach().cpu().numpy()[0]) )

        # break #DEBUG

    end_onnx_inference_batch = time.time()
    print("Total batch tokenization time (in seconds): ", total_batch_tokenization_time)
    print("Total inference time (in seconds): ", total_inference_time)
    print("Total build label time (in seconds): ", total_build_label_time)
    print("Duration ONNX inference (in seconds) with {} and batch tokenization: ".format(tokenizer_str),
          end_onnx_inference_batch - start_onnx_inference_batch,
          (end_onnx_inference_batch - start_onnx_inference_batch) / len(examples))

    return onnx_inference


def get_env_var(varname, default):
    if os.environ.get(varname) != None:
        var = int(os.environ.get(varname))
        print(varname, ':', var)
    else:
        var = default
        print(varname, ':', var, '(Default)')
    return var


# Choose Number of Nodes To Distribute Credentials: e.g. jobarray=0-4, cpu_per_task=20, credentials = 90 (<100)
SLURM_ARRAY_TASK_ID = get_env_var('SLURM_ARRAY_TASK_ID', 0)
SLURM_ARRAY_TASK_COUNT = get_env_var('SLURM_ARRAY_TASK_COUNT', 1)
SLURM_JOB_ID = get_env_var('SLURM_JOB_ID', 1)
print('Hostname:', socket.gethostname())
print('SLURM_ARRAY_TASK_ID', SLURM_ARRAY_TASK_ID)
print('SLURM_ARRAY_TASK_COUNT', SLURM_ARRAY_TASK_COUNT)
# ####################################################################################################################################
# # loading data
# ####################################################################################################################################

path_to_data = args.input_path

print('Load random Tweets:')

start_time = time.time()

final_output_path = args.output_path #e.g. /scratch/spf248/twitter/data/user_timeline/bert_inferrred/MX

if not os.path.exists(os.path.join(final_output_path)):
    print('>>>> directory doesnt exists, creating it')
    os.makedirs(os.path.join(final_output_path))

input_files_list = glob(os.path.join(path_to_data, '*.parquet'))

"""
creating a list of unique file ids assuming this file name structure:
/scratch/spf248/twitter/data/user_timeline/user_timeline_text_preprocessed/part-00000-52fdb0a4-e509-49fe-9f3a-d809594bba7d-c000.snappy.parquet
in this case:
unique_intput_file_id_list will contain 00000-52fdb0a4-e509-49fe-9f3a-d809594bba7d
filename_prefix is /scratch/spf248/twitter/data/user_timeline/user_timeline_text_preprocessed/part-
filename_suffix is -c000.snappy.parquet
"""
unique_intput_file_id_list = [filename.split('part-')[1].split('-c000')[0]
                              for filename in input_files_list]
filename_prefix = input_files_list[0].split('part-')[0]
filename_suffix = input_files_list[0].split('part-')[1].split('-c000')[1]

already_processed_output_files = glob(os.path.join(final_output_path, '*.parquet'))
unique_already_processed_file_id_list = [filename.split('part-')[1].split('-c000')[0]
                              for filename in already_processed_output_files]

if args.resume == 1:
    unique_ids_remaining = list(set(unique_intput_file_id_list) - set(unique_already_processed_file_id_list))
    files_remaining = [filename_prefix+'part-'+filename+'-c000'+filename_suffix for filename in unique_ids_remaining]
    print(files_remaining[:3])
    print(len(files_remaining), len(unique_intput_file_id_list), len(unique_already_processed_file_id_list))
else:
    files_remaining = input_files_list
print('resume', args.resume, len(files_remaining), len(unique_intput_file_id_list),
      len(unique_already_processed_file_id_list))

paths_to_random = list(np.array_split(
        files_remaining,
        SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID]
    )
print('#files in paths_to_random', len(paths_to_random))


if args.method == 0:
    best_model_folders_dict = {
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
            },
            'iter5': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_may26_iter5_7147724_seed-14',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_may26_iter5_7147721_seed-11',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_may26_iter5_7147712_seed-2',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_may26_iter5_7147715_seed-5',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_may26_iter5_7147720_seed-10'
            },
            'iter6': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_may29_iter6_7232047_seed-3',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_may29_iter6_7232051_seed-7',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_may29_iter6_7232051_seed-7',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_may29_iter6_7232045_seed-1',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_may29_iter6_7232059_seed-15'
            },
            'iter7': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_jul19_iter7_8788309_seed-4',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_jul19_iter7_8788311_seed-6',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_jul19_iter7_8788314_seed-9',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_jul19_iter7_8788309_seed-4',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_jul19_iter7_8788315_seed-10'
            },
            'iter8': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_jul22_iter8_8850799_seed-5',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_jul22_iter8_8850801_seed-7',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_jul22_iter8_8850800_seed-6',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_jul22_iter8_8850798_seed-4',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_jul22_iter8_8850802_seed-8'
            }},
        'BR': {'iter0': {
                'lost_job_1mo': 'neuralmind-bert-base-portuguese-cased_feb16_iter0_2843324_seed-12',
                'is_hired_1mo': 'neuralmind-bert-base-portuguese-cased_feb16_iter0_2843317_seed-5',
                'is_unemployed': 'neuralmind-bert-base-portuguese-cased_feb16_iter0_2843317_seed-5',
                'job_offer': 'neuralmind-bert-base-portuguese-cased_feb16_iter0_2843318_seed-6',
                'job_search': 'neuralmind-bert-base-portuguese-cased_feb16_iter0_2843320_seed-8'
            },
            'iter1': {
                'lost_job_1mo': 'neuralmind-bert-base-portuguese-cased_mar12_iter1_3742968_seed-6',
                'is_hired_1mo': 'neuralmind-bert-base-portuguese-cased_mar12_iter1_3742968_seed-6',
                'is_unemployed': 'neuralmind-bert-base-portuguese-cased_mar12_iter1_3742972_seed-10',
                'job_offer': 'neuralmind-bert-base-portuguese-cased_mar12_iter1_3742970_seed-8',
                'job_search': 'neuralmind-bert-base-portuguese-cased_mar12_iter1_3742966_seed-4'},
            'iter2': {
                'lost_job_1mo': 'neuralmind-bert-base-portuguese-cased_mar24_iter2_4173786_seed-10',
                'is_hired_1mo': 'neuralmind-bert-base-portuguese-cased_mar24_iter2_4173783_seed-7',
                'is_unemployed': 'neuralmind-bert-base-portuguese-cased_mar24_iter2_4173787_seed-11',
                'job_offer': 'neuralmind-bert-base-portuguese-cased_mar24_iter2_4173785_seed-9',
                'job_search': 'neuralmind-bert-base-portuguese-cased_mar24_iter2_4173784_seed-8'
            },
            'iter3': {
                'lost_job_1mo': 'neuralmind-bert-base-portuguese-cased_apr1_iter3_4518519_seed-6',
                'is_hired_1mo': 'neuralmind-bert-base-portuguese-cased_apr1_iter3_4518514_seed-1',
                'is_unemployed': 'neuralmind-bert-base-portuguese-cased_apr1_iter3_4518519_seed-6',
                'job_offer': 'neuralmind-bert-base-portuguese-cased_apr1_iter3_4518525_seed-12',
                'job_search': 'neuralmind-bert-base-portuguese-cased_apr1_iter3_4518514_seed-1'},
            'iter4': {
                'lost_job_1mo': 'neuralmind-bert-base-portuguese-cased_apr3_iter4_4677938_seed-6',
                'is_hired_1mo': 'neuralmind-bert-base-portuguese-cased_apr3_iter4_4677933_seed-1',
                'is_unemployed': 'neuralmind-bert-base-portuguese-cased_apr3_iter4_4677934_seed-2',
                'job_offer': 'neuralmind-bert-base-portuguese-cased_apr3_iter4_4677934_seed-2',
                'job_search': 'neuralmind-bert-base-portuguese-cased_apr3_iter4_4677933_seed-1'},
            'iter5': {
                'lost_job_1mo': 'neuralmind-bert-base-portuguese-cased_may17_iter5_6886444_seed-13',
                'is_hired_1mo': 'neuralmind-bert-base-portuguese-cased_may17_iter5_6886442_seed-11',
                'is_unemployed': 'neuralmind-bert-base-portuguese-cased_may17_iter5_6886435_seed-4',
                'job_offer': 'neuralmind-bert-base-portuguese-cased_may17_iter5_6886437_seed-6',
                'job_search': 'neuralmind-bert-base-portuguese-cased_may17_iter5_6886436_seed-5'
            },
            'iter6': {
                'lost_job_1mo': 'neuralmind-bert-base-portuguese-cased_may22_iter6_7053031_seed-4',
                'is_hired_1mo': 'neuralmind-bert-base-portuguese-cased_may22_iter6_7053036_seed-9',
                'is_unemployed': 'neuralmind-bert-base-portuguese-cased_may22_iter6_7053029_seed-2',
                'job_offer': 'neuralmind-bert-base-portuguese-cased_may22_iter6_7053030_seed-3',
                'job_search': 'neuralmind-bert-base-portuguese-cased_may22_iter6_7053034_seed-7'
            },
            'iter7': {
                'lost_job_1mo': 'neuralmind-bert-base-portuguese-cased_may30_iter7_7267580_seed-7',
                'is_hired_1mo': 'neuralmind-bert-base-portuguese-cased_may30_iter7_7267580_seed-7',
                'is_unemployed': 'neuralmind-bert-base-portuguese-cased_may30_iter7_7267580_seed-7',
                'job_offer': 'neuralmind-bert-base-portuguese-cased_may30_iter7_7267588_seed-15',
                'job_search': 'neuralmind-bert-base-portuguese-cased_may30_iter7_7267588_seed-15'
            },
            'iter8': {
                'lost_job_1mo': 'neuralmind-bert-base-portuguese-cased_jun8_iter8_7448808_seed-15',
                'is_hired_1mo': 'neuralmind-bert-base-portuguese-cased_jun8_iter8_7448795_seed-2',
                'is_unemployed': 'neuralmind-bert-base-portuguese-cased_jun8_iter8_7448805_seed-12',
                'job_offer': 'neuralmind-bert-base-portuguese-cased_jun8_iter8_7448800_seed-7',
                'job_search': 'neuralmind-bert-base-portuguese-cased_jun8_iter8_7448794_seed-1'
            },
            'iter9': {
                'lost_job_1mo': 'neuralmind-bert-base-portuguese-cased_jun24_iter9_7985493_seed-11',
                'is_hired_1mo': 'neuralmind-bert-base-portuguese-cased_jun24_iter9_7985486_seed-4',
                'is_unemployed': 'neuralmind-bert-base-portuguese-cased_jun24_iter9_7985495_seed-13',
                'job_offer': 'neuralmind-bert-base-portuguese-cased_jun24_iter9_7985492_seed-10',
                'job_search': 'neuralmind-bert-base-portuguese-cased_jun24_iter9_7985489_seed-7'
            },
        },
        'MX': {
            'iter0': {
                'lost_job_1mo': 'dccuchile-bert-base-spanish-wwm-cased_feb27_iter0_3200976_seed-10',
                'is_hired_1mo': 'dccuchile-bert-base-spanish-wwm-cased_feb27_iter0_3200974_seed-8',
                'is_unemployed': 'dccuchile-bert-base-spanish-wwm-cased_feb27_iter0_3200978_seed-12',
                'job_offer': 'dccuchile-bert-base-spanish-wwm-cased_feb27_iter0_3200968_seed-2',
                'job_search': 'dccuchile-bert-base-spanish-wwm-cased_feb27_iter0_3200967_seed-1'
            },
            'iter1': {
                'lost_job_1mo': 'dccuchile-bert-base-spanish-wwm-cased_mar12_iter1_3737747_seed-8',
                'is_hired_1mo': 'dccuchile-bert-base-spanish-wwm-cased_mar12_iter1_3737745_seed-6',
                'is_unemployed': 'dccuchile-bert-base-spanish-wwm-cased_mar12_iter1_3737741_seed-2',
                'job_offer': 'dccuchile-bert-base-spanish-wwm-cased_mar12_iter1_3737746_seed-7',
                'job_search': 'dccuchile-bert-base-spanish-wwm-cased_mar12_iter1_3737745_seed-6'},
            'iter2': {
                'lost_job_1mo': 'dccuchile-bert-base-spanish-wwm-cased_mar23_iter2_4138955_seed-14',
                'is_hired_1mo': 'dccuchile-bert-base-spanish-wwm-cased_mar23_iter2_4138956_seed-15',
                'is_unemployed': 'dccuchile-bert-base-spanish-wwm-cased_mar23_iter2_4138953_seed-12',
                'job_offer': 'dccuchile-bert-base-spanish-wwm-cased_mar23_iter2_4138951_seed-10',
                'job_search': 'dccuchile-bert-base-spanish-wwm-cased_mar23_iter2_4138943_seed-2'},
            'iter3': {
                'lost_job_1mo': 'dccuchile-bert-base-spanish-wwm-cased_mar30_iter3_4375824_seed-2',
                'is_hired_1mo': 'dccuchile-bert-base-spanish-wwm-cased_mar30_iter3_4375831_seed-9',
                'is_unemployed': 'dccuchile-bert-base-spanish-wwm-cased_mar30_iter3_4375832_seed-10',
                'job_offer': 'dccuchile-bert-base-spanish-wwm-cased_mar30_iter3_4375832_seed-10',
                'job_search': 'dccuchile-bert-base-spanish-wwm-cased_mar30_iter3_4375830_seed-8'},
            'iter4': {
                'lost_job_1mo': 'dccuchile-bert-base-spanish-wwm-cased_apr2_iter4_4597713_seed-4',
                'is_hired_1mo': 'dccuchile-bert-base-spanish-wwm-cased_apr2_iter4_4597718_seed-9',
                'is_unemployed': 'dccuchile-bert-base-spanish-wwm-cased_apr2_iter4_4597718_seed-9',
                'job_offer': 'dccuchile-bert-base-spanish-wwm-cased_apr2_iter4_4597712_seed-3',
                'job_search': 'dccuchile-bert-base-spanish-wwm-cased_apr2_iter4_4597710_seed-1'},
            'iter6': {
                'lost_job_1mo': 'dccuchile-bert-base-spanish-wwm-cased_may25_iter6_7125251_seed-4',
                'is_hired_1mo': 'dccuchile-bert-base-spanish-wwm-cased_may25_iter6_7125254_seed-7',
                'is_unemployed': 'dccuchile-bert-base-spanish-wwm-cased_may25_iter6_7125255_seed-8',
                'job_offer': 'dccuchile-bert-base-spanish-wwm-cased_may25_iter6_7125252_seed-5',
                'job_search': 'dccuchile-bert-base-spanish-wwm-cased_may25_iter6_7125251_seed-4'
            },
            'iter7': {
                'lost_job_1mo': 'dccuchile-bert-base-spanish-wwm-cased_may31_iter7_7272629_seed-7',
                'is_hired_1mo': 'dccuchile-bert-base-spanish-wwm-cased_may31_iter7_7272633_seed-11',
                'is_unemployed': 'dccuchile-bert-base-spanish-wwm-cased_may31_iter7_7272630_seed-8',
                'job_offer': 'dccuchile-bert-base-spanish-wwm-cased_may31_iter7_7272629_seed-7',
                'job_search': 'dccuchile-bert-base-spanish-wwm-cased_may31_iter7_7272634_seed-12'
            },
            'iter8': {
                'lost_job_1mo': 'dccuchile-bert-base-spanish-wwm-cased_jun5_iter8_7383859_seed-1',
                'is_hired_1mo': 'dccuchile-bert-base-spanish-wwm-cased_jun5_iter8_7383871_seed-13',
                'is_unemployed': 'dccuchile-bert-base-spanish-wwm-cased_jun5_iter8_7383867_seed-9',
                'job_offer': 'dccuchile-bert-base-spanish-wwm-cased_jun5_iter8_7383867_seed-9',
                'job_search': 'dccuchile-bert-base-spanish-wwm-cased_jun5_iter8_7383866_seed-8'
            },
            'iter9': {
                'lost_job_1mo': 'dccuchile-bert-base-spanish-wwm-cased_jun6_iter9_7408605_seed-1',
                'is_hired_1mo': 'dccuchile-bert-base-spanish-wwm-cased_jun6_iter9_7408615_seed-11',
                'is_unemployed': 'dccuchile-bert-base-spanish-wwm-cased_jun6_iter9_7408607_seed-3',
                'job_offer': 'dccuchile-bert-base-spanish-wwm-cased_jun6_iter9_7408614_seed-10',
                'job_search': 'dccuchile-bert-base-spanish-wwm-cased_jun6_iter9_7408609_seed-5'
            },
            'iter10': {
                'lost_job_1mo': 'dccuchile-bert-base-spanish-wwm-cased_jun30_iter10_8222598_seed-12',
                'is_hired_1mo': 'dccuchile-bert-base-spanish-wwm-cased_jun30_iter10_8222593_seed-7',
                'is_unemployed': 'dccuchile-bert-base-spanish-wwm-cased_jun30_iter10_8222591_seed-5',
                'job_offer': 'dccuchile-bert-base-spanish-wwm-cased_jun30_iter10_8222587_seed-1',
                'job_search': 'dccuchile-bert-base-spanish-wwm-cased_jun30_iter10_8222592_seed-6'
            },
        }
    }

tweets_random = pd.DataFrame()

TOTAL_NUM_TWEETS = 0
for file in paths_to_random:
    print(file)
    filename_without_extension = os.path.splitext(os.path.splitext(file.split('/')[-1])[0])[0]
    print('filename_without_extension')


    tweets_random = pd.read_parquet(file)[['tweet_id', 'text']]
    print(tweets_random.shape)

    # tweets_random = tweets_random.head(10) #DEBUG

    print('load random sample:', str(time.time() - start_time), 'seconds')
    print(tweets_random.shape)


    if args.drop_duplicates:
        print('dropping duplicates:')
        start_time = time.time()
        tweets_random = tweets_random.drop_duplicates('text')
        print('drop duplicates:', str(time.time() - start_time), 'seconds')
        print(tweets_random.shape)

    start_time = time.time()
    print('converting to list')
    examples = tweets_random.text.values.tolist()
    # examples = examples[0] #DEBUG
    TOTAL_NUM_TWEETS = TOTAL_NUM_TWEETS + len(examples)

    print('convert to list:', str(time.time() - start_time), 'seconds')
    all_predictions_random_df_list = []
    for column in ["is_unemployed", "lost_job_1mo", "job_search", "is_hired_1mo", "job_offer"]:
        print('\n\n!!!!!column', column)
        loop_start = time.time()
        best_model_folder = best_model_folders_dict[args.country_code][f'iter{str(args.iteration_number)}'][column]
        model_path = os.path.join('/scratch/mt4493/twitter_labor/trained_models', args.country_code, best_model_folder,
        column, 'models', 'best_model')

        print(model_path)
        onnx_path = os.path.join(model_path, 'onnx')
        print(onnx_path)

        ####################################################################################################################################
        # TOKENIZATION and INFERENCE
        ####################################################################################################################################
        print('Predictions of random Tweets:')
        start_time = time.time()
        onnx_labels = inference(os.path.join(onnx_path, 'converted-optimized-quantized.onnx'),
                                model_path,
                                examples)

        print('time taken:', str(time.time() - start_time), 'seconds')
        print('per tweet:', (time.time() - start_time) / tweets_random.shape[0], 'seconds')

        ####################################################################################################################################
        # SAVING
        ####################################################################################################################################
        print('Save Predictions of random Tweets:')
        start_time = time.time()

        # create dataframe containing tweet id and probabilities
        predictions_random_df = pd.DataFrame(data=onnx_labels, columns=['first', 'second'])
        predictions_random_df = predictions_random_df.set_index(tweets_random.tweet_id)
        # reformat dataframe
        predictions_random_df = predictions_random_df[['second']]
        # predictions_random_df.columns = ['score']
        predictions_random_df.columns = [column]
        print(predictions_random_df.head())

        all_predictions_random_df_list.append(predictions_random_df)

        # break  # DEBUG column

    all_columns_df = reduce(lambda x,y: pd.merge(x , y, left_on=['tweet_id'], right_on=['tweet_id'] ,how='inner'),
                            all_predictions_random_df_list
                            )

    print('!!all_columns_df', all_columns_df.head())
    print('!!shapes', all_columns_df.shape, [df.shape for df in all_predictions_random_df_list])
    all_columns_df.to_parquet(
        os.path.join(final_output_path,
                     filename_without_extension + str(getpass.getuser()) + '_random' + '-' + str(SLURM_ARRAY_TASK_ID)
                     + '.parquet'))

    print('saved to:',
          # column,
          SLURM_ARRAY_TASK_ID,
          SLURM_JOB_ID,
          SLURM_ARRAY_TASK_COUNT,
          filename_without_extension,
          os.path.join(final_output_path,
                                      filename_without_extension + str(getpass.getuser()) + '_random' + '-' + str(SLURM_ARRAY_TASK_ID) + '.parquet'),
          str(time.time() - start_time)
        )

    print('>>>>> completed', filename_without_extension)

    print('save time taken:', str(time.time() - start_time), 'seconds')

    print('file loop:', filename_without_extension, str(time.time() - loop_start), 'seconds', (time.time() -
                                                                                                  loop_start) / len(examples))
    # break #DEBUG parquet file


print('full loop:', str(time.time() - global_start), 'seconds',
      (time.time() - global_start) / TOTAL_NUM_TWEETS)

print('>>done')
