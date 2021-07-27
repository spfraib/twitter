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


args = parser.parse_args()

print(args)


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

paths_to_random = list(np.array_split(
    glob(os.path.join(path_to_data, '*.parquet')),
    SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID])
print('#files:', len(paths_to_random))

tweets_random = pd.DataFrame()
for file in paths_to_random:
    print(file)
    tweets_random = pd.concat([tweets_random, pd.read_parquet(file)[['tweet_id', 'text']]])
    print(tweets_random.shape)

print('load random sample:', str(time.time() - start_time), 'seconds')
print(tweets_random.shape)

print('dropping duplicates:')
# random contains 7.3G of data!!
start_time = time.time()
tweets_random = tweets_random.drop_duplicates('text')
print('drop duplicates:', str(time.time() - start_time), 'seconds')
print(tweets_random.shape)

start_time = time.time()
print('converting to list')
examples = tweets_random.text.values.tolist()

print('convert to list:', str(time.time() - start_time), 'seconds')

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
            },
        },
        'BR': {
            'iter0': {
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
            'iter5': {
                'lost_job_1mo': 'dccuchile-bert-base-spanish-wwm-cased_may17_iter5_6886418_seed-2',
                'is_hired_1mo': 'dccuchile-bert-base-spanish-wwm-cased_may17_iter5_6886419_seed-3',
                'is_unemployed': 'dccuchile-bert-base-spanish-wwm-cased_may17_iter5_6886422_seed-6',
                'job_offer': 'dccuchile-bert-base-spanish-wwm-cased_may17_iter5_6886421_seed-5',
                'job_search': 'dccuchile-bert-base-spanish-wwm-cased_may17_iter5_6886423_seed-7'
            },
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
            'iter11': {
                'lost_job_1mo': 'dccuchile-bert-base-spanish-wwm-cased_jul20_iter11_8804820_seed-6',
                'is_hired_1mo': 'dccuchile-bert-base-spanish-wwm-cased_jul20_iter11_8804823_seed-9',
                'is_unemployed': 'dccuchile-bert-base-spanish-wwm-cased_jul20_iter11_8804817_seed-3',
                'job_offer': 'dccuchile-bert-base-spanish-wwm-cased_jul20_iter11_8804823_seed-9',
                'job_search': 'dccuchile-bert-base-spanish-wwm-cased_jul20_iter11_8804819_seed-5'
            },
            'iter12': {
                'lost_job_1mo': 'dccuchile-bert-base-spanish-wwm-cased_jul27_iter12_8966158_seed-4',
                'is_hired_1mo': 'dccuchile-bert-base-spanish-wwm-cased_jul27_iter12_8966159_seed-5',
                'is_unemployed': 'dccuchile-bert-base-spanish-wwm-cased_jul27_iter12_8966162_seed-8',
                'job_offer': 'dccuchile-bert-base-spanish-wwm-cased_jul27_iter12_8966155_seed-1',
                'job_search': 'dccuchile-bert-base-spanish-wwm-cased_jul27_iter12_8966167_seed-13'
            },
        }
    }

elif args.method == 1:
    best_model_folders_dict = {
        'US': {
            'iter1': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_apr19_iter1_adaptive_5598877_seed-5',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_apr19_iter1_adaptive_5598877_seed-5',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_apr19_iter1_adaptive_5598883_seed-11',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_apr19_iter1_adaptive_5598880_seed-8',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_apr19_iter1_adaptive_5598880_seed-8'},
            'iter2': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_apr25_iter2_adaptive_5972290_seed-14',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_apr25_iter2_adaptive_5972289_seed-13',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_apr25_iter2_adaptive_5972286_seed-10',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_apr25_iter2_adaptive_5972278_seed-2',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_apr25_iter2_adaptive_5972280_seed-4'
            },
            'iter3': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_apr26_iter3_adaptive_5997887_seed-6',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_apr26_iter3_adaptive_5997886_seed-5',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_apr26_iter3_adaptive_5997886_seed-5',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_apr26_iter3_adaptive_5997890_seed-9',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_apr26_iter3_adaptive_5997893_seed-12'
            },
            'iter4': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_apr27_iter4_adaptive_6026892_seed-10',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_apr27_iter4_adaptive_6026884_seed-2',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_apr27_iter4_adaptive_6026889_seed-7',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_apr27_iter4_adaptive_6026884_seed-2',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_apr27_iter4_adaptive_6026894_seed-12'
            },
            'iter5': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_may12_iter5_adaptive_6739858_seed-6',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_may12_iter5_adaptive_6739863_seed-11',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_may12_iter5_adaptive_6739854_seed-2',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_may12_iter5_adaptive_6739861_seed-9',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_may12_iter5_adaptive_6739853_seed-1'
            },
            'iter6': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_may28_iter6_adaptive_7206891_seed-2',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_may28_iter6_adaptive_7206893_seed-4',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_may28_iter6_adaptive_7206895_seed-6',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_may28_iter6_adaptive_7206894_seed-5',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_may28_iter6_adaptive_7206904_seed-15'
            }
        }}

elif args.method == 2:
    best_model_folders_dict = {
        'US': {
            'iter1': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_apr30_iter1_uncertainty_6196561_seed-11',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_apr30_iter1_uncertainty_6196555_seed-5',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_apr30_iter1_uncertainty_6196561_seed-11',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_apr30_iter1_uncertainty_6196560_seed-10',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_apr30_iter1_uncertainty_6196553_seed-3'},
            'iter2': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_may1_iter2_uncertainty_6244850_seed-11',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_may1_iter2_uncertainty_6244843_seed-4',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_may1_iter2_uncertainty_6244841_seed-2',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_may1_iter2_uncertainty_6244840_seed-1',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_may1_iter2_uncertainty_6244850_seed-11'
            },
            'iter3': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_may2_iter3_uncertainty_6314074_seed-4',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_may2_iter3_uncertainty_6314072_seed-2',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_may2_iter3_uncertainty_6314083_seed-13',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_may2_iter3_uncertainty_6314080_seed-10',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_may2_iter3_uncertainty_6314071_seed-1'
            },
            'iter4': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_may3_iter4_uncertainty_6349919_seed-1',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_may3_iter4_uncertainty_6378411_seed-13',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_may3_iter4_uncertainty_6378403_seed-5',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_may3_iter4_uncertainty_6378407_seed-9',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_may3_iter4_uncertainty_6378409_seed-11'
            },
            'iter5': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_may11_iter5_uncertainty_6711052_seed-2',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_may11_iter5_uncertainty_6711053_seed-3',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_may11_iter5_uncertainty_6711059_seed-9',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_may11_iter5_uncertainty_6711055_seed-5',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_may11_iter5_uncertainty_6711054_seed-4'
            }
        }}

elif args.method == 3:
    best_model_folders_dict = {
        'US': {
            'iter1': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_may6_iter1_uncertainty_uncalibrated_6471039_seed-4',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_may6_iter1_uncertainty_uncalibrated_6471042_seed-7',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_may6_iter1_uncertainty_uncalibrated_6471047_seed-12',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_may6_iter1_uncertainty_uncalibrated_6471036_seed-1',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_may6_iter1_uncertainty_uncalibrated_6471048_seed-13'},
            'iter2': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_may7_iter2_uncertainty_uncalibrated_6518196_seed-10',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_may7_iter2_uncertainty_uncalibrated_6518200_seed-14',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_may7_iter2_uncertainty_uncalibrated_6518187_seed-1',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_may7_iter2_uncertainty_uncalibrated_6518197_seed-11',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_may7_iter2_uncertainty_uncalibrated_6518187_seed-1'
            },
            'iter3': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_may8_iter3_uncertainty_uncalibrated_6583469_seed-5',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_may8_iter3_uncertainty_uncalibrated_6583465_seed-1',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_may8_iter3_uncertainty_uncalibrated_6583472_seed-8',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_may8_iter3_uncertainty_uncalibrated_6583478_seed-14',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_may8_iter3_uncertainty_uncalibrated_6583472_seed-8'
            },
            'iter4': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_may10_iter4_uncertainty_uncalibrated_6653463_seed-2',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_may10_iter4_uncertainty_uncalibrated_6653473_seed-12',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_may10_iter4_uncertainty_uncalibrated_6653473_seed-12',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_may10_iter4_uncertainty_uncalibrated_6653464_seed-3',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_may10_iter4_uncertainty_uncalibrated_6653472_seed-11'
            },
            'iter5': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_may12_iter5_uncertainty_uncalibrated_6737085_seed-12',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_may12_iter5_uncertainty_uncalibrated_6737082_seed-9',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_may12_iter5_uncertainty_uncalibrated_6737074_seed-1',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_may12_iter5_uncertainty_uncalibrated_6737077_seed-4',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_may12_iter5_uncertainty_uncalibrated_6737086_seed-13'
            }
        }}

for column in ["is_unemployed", "lost_job_1mo", "job_search", "is_hired_1mo", "job_offer"]:
    print('\n\n!!!!!', column)
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
    final_output_path = args.output_path
    if not os.path.exists(os.path.join(final_output_path, column)):
        print('>>>> directory doesnt exists, creating it')
        os.makedirs(os.path.join(final_output_path, column))
    # create dataframe containing tweet id and probabilities
    predictions_random_df = pd.DataFrame(data=onnx_labels, columns=['first', 'second'])
    predictions_random_df = predictions_random_df.set_index(tweets_random.tweet_id)
    # reformat dataframe
    predictions_random_df = predictions_random_df[['second']]
    predictions_random_df.columns = ['score']

    print(predictions_random_df.head())
    predictions_random_df.to_parquet(
    os.path.join(final_output_path, column,
                 str(getpass.getuser()) + '_random' + '-' + str(SLURM_ARRAY_TASK_ID) + '.parquet'))

    print('saved to:\n', os.path.join(final_output_path, column,
                                      str(getpass.getuser()) + '_random' + '-' + str(SLURM_ARRAY_TASK_ID) + '.parquet'),
    'saved')

    print('save time taken:', str(time.time() - start_time), 'seconds')

    print('full loop:', str(time.time() - loop_start), 'seconds', (time.time() - loop_start) / len(examples))
