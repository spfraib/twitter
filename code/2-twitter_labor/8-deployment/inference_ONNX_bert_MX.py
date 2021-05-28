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
parser.add_argument("--debug_mode", type=bool, help="fast debug mode", default=True)
parser.add_argument("--drop_duplicates", type=bool, help="drop duplicated tweets from parquet files", default=False)


args = parser.parse_args()

print(args)

DEBUG_MODE = args.debug_mode

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
                                    SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID]
                       )
print('#files:', len(paths_to_random))


if args.method == 0:
    best_model_folders_dict = {
        'US': {},
        'BR': {},
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
            }
        }
    }

tweets_random = pd.DataFrame()

for file in paths_to_random:
    print(file)
    filename_without_extension = os.path.splitext(os.path.splitext(file.split('/')[-1])[0])[0]

    tweets_random = pd.concat([tweets_random, pd.read_parquet(file)[['tweet_id', 'text']]])
    print(tweets_random.shape)

    print('load random sample:', str(time.time() - start_time), 'seconds')
    print(tweets_random.shape)


    if args.drop_duplicates:
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

    # for column in ["is_unemployed", "lost_job_1mo", "job_search", "is_hired_1mo", "job_offer"]:
    #     print('\n\n!!!!!', column)
    #     loop_start = time.time()
    #     best_model_folder = best_model_folders_dict[args.country_code][f'iter{str(args.iteration_number)}'][column]
    #     model_path = os.path.join('/scratch/mt4493/twitter_labor/trained_models', args.country_code, best_model_folder,
    #     column, 'models', 'best_model')
    #
    #     print(model_path)
    #     onnx_path = os.path.join(model_path, 'onnx')
    #     print(onnx_path)
    #
    #     ####################################################################################################################################
    #     # TOKENIZATION and INFERENCE
    #     ####################################################################################################################################
    #     print('Predictions of random Tweets:')
    #     start_time = time.time()
    #     onnx_labels = inference(os.path.join(onnx_path, 'converted-optimized-quantized.onnx'),
    #     model_path,
    #     examples)
    #
    #     print('time taken:', str(time.time() - start_time), 'seconds')
    #     print('per tweet:', (time.time() - start_time) / tweets_random.shape[0], 'seconds')
    #
    #     ####################################################################################################################################
    #     # SAVING
    #     ####################################################################################################################################
    #     print('Save Predictions of random Tweets:')
    #     start_time = time.time()
    #     final_output_path = args.output_path #e.g. /scratch/spf248/twitter/data/user_timeline/bert_inferrred/MX
    #     if not os.path.exists(os.path.join(final_output_path, column)):
    #         print('>>>> directory doesnt exists, creating it')
    #         os.makedirs(os.path.join(final_output_path, column))
    #     # create dataframe containing tweet id and probabilities
    #     predictions_random_df = pd.DataFrame(data=onnx_labels, columns=['first', 'second'])
    #     predictions_random_df = predictions_random_df.set_index(tweets_random.tweet_id)
    #     # reformat dataframe
    #     predictions_random_df = predictions_random_df[['second']]
    #     predictions_random_df.columns = ['score']
    #
    #     print(predictions_random_df.head())
    #     predictions_random_df.to_parquet(
    #     os.path.join(final_output_path, column,
    #                  filename_without_extension,
    #                  str(getpass.getuser()) + '_random' + '-' + str(SLURM_ARRAY_TASK_ID) + '.parquet'))
    #
    #     print('saved to:\n', os.path.join(final_output_path, column,
    #                                       str(getpass.getuser()) + '_random' + '-' + str(SLURM_ARRAY_TASK_ID) + '.parquet'),
    #     'saved')
    #
    #     print('save time taken:', str(time.time() - start_time), 'seconds')
    #
    #     print('full loop:', str(time.time() - loop_start), 'seconds', (time.time() - loop_start) / len(examples))
    #
    #     break #DEBUG column

    break #DEBUG parquet file

print('done')