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

# logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#                     datefmt='%m/%d/%Y %H:%M:%S',
#                     level=logging.INFO)
# logger = logging.getLogger(__name__)
# print('libs loaded')

# parser = argparse.ArgumentParser()

# parser.add_argument("--input_path", type=str, help="path to input data")
# parser.add_argument("--output_path", type=str, help="path where inference csv is saved")
# parser.add_argument("--country_code", type=str, help="path where inference csv is saved")
# parser.add_argument("--iteration_number", type=int)
# parser.add_argument("--method", type=int)


# args = parser.parse_args()

# print(args)


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


# # Choose Number of Nodes To Distribute Credentials: e.g. jobarray=0-4, cpu_per_task=20, credentials = 90 (<100)
# SLURM_ARRAY_TASK_ID = get_env_var('SLURM_ARRAY_TASK_ID', 0)
# SLURM_ARRAY_TASK_COUNT = get_env_var('SLURM_ARRAY_TASK_COUNT', 1)
# SLURM_JOB_ID = get_env_var('SLURM_JOB_ID', 1)
# print('Hostname:', socket.gethostname())
# print('SLURM_ARRAY_TASK_ID', SLURM_ARRAY_TASK_ID)
# print('SLURM_ARRAY_TASK_COUNT', SLURM_ARRAY_TASK_COUNT)
# ####################################################################################################################################
# # loading data
# ####################################################################################################################################

# path_to_data = args.input_path

print('Load random Tweets:')

start_time = time.time()

# paths_to_random = list(np.array_split(
#     glob(os.path.join(path_to_data, '*.parquet')),
#     SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID])
# print('#files:', len(paths_to_random))

# tweets_random = pd.DataFrame()
# for file in paths_to_random:
#     print(file)
#     tweets_random = pd.concat([tweets_random, pd.read_parquet(file)[['tweet_id', 'text']]])
#     print(tweets_random.shape)

# print('load random sample:', str(time.time() - start_time), 'seconds')
# print(tweets_random.shape)

# print('dropping duplicates:')
# # random contains 7.3G of data!!
# start_time = time.time()
# tweets_random = tweets_random.drop_duplicates('text')
# print('drop duplicates:', str(time.time() - start_time), 'seconds')
# print(tweets_random.shape)

# start_time = time.time()
# print('converting to list')
# examples = tweets_random.text.values.tolist()

# print('convert to list:', str(time.time() - start_time), 'seconds')

path_to_data = '/scratch/mt4493/twitter_labor/twitter-labor-data/data/random_samples/random_samples_splitted/US/test'
tweets_random = pd.DataFrame()
# print('path_to_data', path_to_data)
for file in os.listdir(path_to_data):
    print('reading', file)
    tweets_random = pd.concat([tweets_random,
                               pd.read_parquet(path_to_data+'/'+file)[['tweet_id', 'text']]])
    break

# print('input shape', tweets_random.shape)
NUM_TWEETS = 1000
BATCH_SIZE = 1
MODEL_TYPE = 'manu_current'
tweets_random = tweets_random.head(NUM_TWEETS)

tweets_random = tweets_random.drop_duplicates('text')

start_time = time.time()
# print('converting to list')
examples = tweets_random.text.values.tolist()

column = 'job_search'

# print('\n\n!!!!!', column)
# loop_start = time.time()
# best_model_folder = best_model_folders_dict[args.country_code][f'iter{str(args.iteration_number)}'][column]
# model_path = os.path.join('/scratch/mt4493/twitter_labor/trained_models', args.country_code, best_model_folder,
# column, 'models', 'best_model')

onnx_model_path = '/scratch/mt4493/twitter_labor/trained_models/US/DeepPavlov-bert-base-cased-conversational_mar1_iter4_3297484_seed-10/job_search/models/best_model/'
# print(model_path)
onnx_path = os.path.join(onnx_model_path, 'onnx')
# print(onnx_path)

# model_type = 'converted.onnx'
# model_type = 'converted-optimized.onnx'
model_type = 'converted-optimized-quantized.onnx'

    ####################################################################################################################################
    # TOKENIZATION and INFERENCE
    ####################################################################################################################################

for REPLICATION in range(5):
    print('Predictions of random Tweets:')
    start_time = time.time()
    onnx_labels = inference(os.path.join(onnx_path, model_type),
                                        onnx_model_path,
                                        examples)

    onnx_total_time = float(str(time.time() - start_time))
    onnx_per_tweet = onnx_total_time / tweets_random.shape[0]

    print('time taken:', str(time.time() - start_time), 'seconds')
    print('per tweet:', (time.time() - start_time) / tweets_random.shape[0], 'seconds')



    # ####################################################################################################################################
    # # SAVING
    # ####################################################################################################################################
    # print('Save Predictions of random Tweets:')
    # start_time = time.time()
    # final_output_path = args.output_path
    # if not os.path.exists(os.path.join(final_output_path, column)):
    #     print('>>>> directory doesnt exists, creating it')
    #     os.makedirs(os.path.join(final_output_path, column))
    # # create dataframe containing tweet id and probabilities
    # predictions_random_df = pd.DataFrame(data=onnx_labels, columns=['first', 'second'])
    # predictions_random_df = predictions_random_df.set_index(tweets_random.tweet_id)
    # # reformat dataframe
    # predictions_random_df = predictions_random_df[['second']]
    # predictions_random_df.columns = ['score']

    # print(predictions_random_df.head())
    # predictions_random_df.to_parquet(
    # os.path.join(final_output_path, column,
    #              str(getpass.getuser()) + '_random' + '-' + str(SLURM_ARRAY_TASK_ID) + '.parquet'))

    # print('saved to:\n', os.path.join(final_output_path, column,
    #                                   str(getpass.getuser()) + '_random' + '-' + str(SLURM_ARRAY_TASK_ID) + '.parquet'),
    # 'saved')

    # print('save time taken:', str(time.time() - start_time), 'seconds')

    # print('full loop:', str(time.time() - loop_start), 'seconds', (time.time() - loop_start) / len(examples))


    # ####################################################################################################################################
    # # CALCULATIONS
    # ####################################################################################################################################
    # print('Save Predictions of random Tweets:')
    # start_time = time.time()
    final_output_path = '/scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/4-inference_200M/bert_models/dhaval_inference_test/replication_output_data'
    # final_output_path = '/Users/dval/work_temp/twitter_from_nyu/output'
    if not os.path.exists(os.path.join(final_output_path, column)):
        # print('>>>> directory doesnt exists, creating it')
        os.makedirs(os.path.join(final_output_path, column))
    # create dataframe containing tweet id and probabilities
    onnx_predictions_random_df = pd.DataFrame(data=onnx_labels, columns=['onnx_score_not_relevant', 'onnx_score'])
    onnx_predictions_random_df = onnx_predictions_random_df.set_index(tweets_random.tweet_id)
    onnx_predictions_random_df['tweet_id'] = onnx_predictions_random_df.index
    onnx_predictions_random_df = onnx_predictions_random_df.reset_index(drop=True)
    onnx_predictions_random_df = onnx_predictions_random_df[['tweet_id', 'onnx_score']]
    onnx_predictions_random_df['onnx_time_per_tweet'] = onnx_per_tweet
    onnx_predictions_random_df['num_tweets'] = NUM_TWEETS
    onnx_predictions_random_df['onnx_batchsize'] = BATCH_SIZE
    onnx_predictions_random_df['onnx_model_type'] = MODEL_TYPE
    onnx_predictions_random_df['device'] = ort.get_device()

    onnx_predictions_random_df.to_csv(
    # merged.to_csv(
                os.path.join(final_output_path, column,
                 str(getpass.getuser()) + '_random' + '-' +
                                   str(MODEL_TYPE) + '-' +
                                   'bs-' + str(BATCH_SIZE) + '-' +
                                   'rep-' + str(REPLICATION) +
                                 '.csv'))

    print('saved to:\n', os.path.join(final_output_path, column,
      str(getpass.getuser()) + '_random' + '-' +
                                  str(MODEL_TYPE) + '-' +
                                  'bs-' + str(BATCH_SIZE) + '-' +
                                  'rep-' + str(REPLICATION) +
                                  '.csv'))
          
#     break
