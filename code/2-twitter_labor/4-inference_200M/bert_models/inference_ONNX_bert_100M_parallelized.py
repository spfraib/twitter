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
import multiprocessing as mp
from functools import partial

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

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

def run_inference(ort_session, tokens):
    ort_outs = ort_session.run(None, tokens)
    torch_onnx_output = torch.tensor(ort_outs[0], dtype=torch.float32)
    onnx_logits = F.softmax(torch_onnx_output, dim=1)
    return onnx_logits.detach().cpu().numpy()[0].tolist()

def inference(onnx_model, model_dir, examples):
    quantized_str = ''
    if 'quantized' in onnx_model:
        quantized_str = 'quantized'
    # onnx session
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 1
    #options.inter_op_num_threads = multiprocessing.cpu_count()

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
    tokens_dict = tokenizer.batch_encode_plus(examples, max_length=128)
    tokens_dict_list = [dict(zip(tokens_dict, t)) for t in zip(*tokens_dict.values())]
    with mp.Pool(mp.cpu_count()) as pool:
        run_inference_on_tokens = partial(run_inference, ort_session=ort_session)
        onnx_inference = pool.map(run_inference_on_tokens, tokens_dict_list)
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
print('Number of CPUs per task:', mp.cpu_count())
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

best_model_folders_dict = {'iter0': { 'US': {
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
}}}

for column in ["is_unemployed", "lost_job_1mo", "job_search", "is_hired_1mo", "job_offer"]:

    print('\n\n!!!!!', column)
    loop_start = time.time()
    best_model_folder = best_model_folders_dict[f'iter{str(args.iteration_number)}'][args.country_code][column]
    model_path = os.path.join('/scratch/mt4493/twitter_labor/trained_models', args.country_code, best_model_folder, column, 'models', 'best_model')

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

