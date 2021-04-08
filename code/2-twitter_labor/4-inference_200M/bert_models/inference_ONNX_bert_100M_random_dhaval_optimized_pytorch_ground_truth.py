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
import json
from simpletransformers.classification import ClassificationModel
from scipy.special import softmax

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
# print('libs loaded')

parser = argparse.ArgumentParser()

parser.add_argument("--input_path", type=str, help="path to input data")
parser.add_argument("--output_path", type=str, help="path where inference csv is saved")
parser.add_argument("--country_code", type=str, help="path where inference csv is saved")
parser.add_argument("--batchsize", type=int, help="batch size for inference")
parser.add_argument("--iteration_number", type=int)

args = parser.parse_args()

# print(args)
def read_json(filename: str):
    with open(filename) as f_in:
        return json.load(f_in)


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


NUM_TWEETS = 100
# BATCH_SIZE = 1
BATCH_SIZE = int(args.batchsize)
print('BATCH_SIZE', BATCH_SIZE)
NUM_BATCHES = int(np.ceil( NUM_TWEETS/BATCH_SIZE ))

# Splitting a list into N parts of approximately equal length
def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out

def inference(model, examples):
    # quantized_str = ''
    # if 'quantized' in onnx_model:
    #     quantized_str = 'quantized'
    # onnx_inference = []
    # #     pytorch_inference = []
    # # onnx session
    # options = ort.SessionOptions()
    # options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # options.intra_op_num_threads = 16 # does not seem to make a difference, always parallelized
    # # options.inter_op_num_threads = multiprocessing.cpu_count()
    #
    # # print(onnx_model)
    # ort_session = ort.InferenceSession(onnx_model, options)

    # pytorch pretrained model and tokenizer
    # if 'bertweet' in onnx_model:
    #     tokenizer = AutoTokenizer.from_pretrained(model_dir, normalization=True)
    # else:
    #     tokenizer = AutoTokenizer.from_pretrained(model_dir)
    #
    # tokenizer_str = "TokenizerFast"
    #
    # # print("**************** {} ONNX inference with batch tokenization and with {} tokenizer****************".format(
    # #     quantized_str, tokenizer_str))
    # start_batch_tokenization = time.time()
    # tokens_dict = tokenizer.batch_encode_plus(examples, max_length=128)
    # token0 = get_tokens(tokens_dict, 0)
    #
    # examples_chunks_list = chunkIt(examples, NUM_BATCHES)
    # tokens_dict_list = [tokenizer.batch_encode_plus(chunk, padding='longest') for chunk in examples_chunks_list]
    # # tokens_dict_list = [tokenizer.batch_encode_plus(chunk, max_length=128) for chunk in examples_chunks_list]
    #
    # minibatches_list = []
    # for i, token_batch in enumerate(tokens_dict_list):
    #     minibatch = {}
    #     number_examples_in_this_batch = len(token_batch['input_ids'])
    #     minibatch['input_ids'] = np.stack((
    #                                 [get_tokens(token_batch, i)['input_ids'][0] for i in range(number_examples_in_this_batch)]
    #                                 ), axis=0)
    #     minibatch['token_type_ids'] = np.stack((
    #                                 [get_tokens(token_batch, i)['token_type_ids'][0] for i in range(number_examples_in_this_batch)]
    #                                 ), axis=0)
    #     minibatch['attention_mask'] = np.stack((
    #                                 [get_tokens(token_batch, i)['attention_mask'][0] for i in range(number_examples_in_this_batch)]
    #                                 ), axis=0)
    #     # print('y')
    #     minibatches_list.append(minibatch)
    #
    # # tokens_dict = tokenizer.batch_encode_plus(examples, padding='longest')
    # total_batch_tokenization_time = time.time() - start_batch_tokenization
    # total_inference_time = 0
    # total_build_label_time = 0
    # start_onnx_inference_batch = time.time()

    # for i, example in enumerate(examples):
    # for i, minibatch in enumerate(minibatches_list):
    """
    Onnx inference with batch tokenization
    """
    # onnx_inference = []

    # if i % 100 == 0:
    #     print(i, '/', NUM_BATCHES)

    # tokens = get_tokens(tokens_dict, i)
    # inference
    start_inference = time.time()
    # ort_outs = ort_session.run(None, tokens)
    # ort_outs = ort_session.run(None, minibatch)
    predictions, raw_outputs = model.predict( examples )
    # predictions, raw_outputs = model.predict( [example] )
    scores = np.array([softmax(element)[1] for element in raw_outputs])
    print(examples)
    print('raw_outputs', raw_outputs)
    print('predictions', predictions)
    print('scores', scores)
    # total_inference_time = total_inference_time + (time.time() - start_inference)
    # build label
    # start_build_label = time.time()
    # torch_onnx_output = torch.tensor(ort_outs[0], dtype=torch.float32)
    # onnx_logits = F.softmax(torch_onnx_output, dim=1)
    # logits_label = torch.argmax(onnx_logits, dim=1)
    # label = logits_label.detach().cpu().numpy()
    #         onnx_inference.append(label[0])
    # onnx_inference.append(onnx_logits.detach().cpu().numpy().tolist())
    # print(scores)

    # TODO might be able to make this faster by using arrays with pre-defined size isntead of mutating lists like this
    # onnx_inference = onnx_inference + scores
    # onnx_inference = onnx_inference + onnx_logits.detach().cpu().numpy().tolist()
    # onnx_inference.append(onnx_logits.detach().cpu().numpy()[0].tolist())
    # total_build_label_time = total_build_label_time + (time.time() - start_build_label)
#         print(i, label[0], onnx_logits.detach().cpu().numpy()[0].tolist(), type(onnx_logits.detach().cpu().numpy()[0]) )
#     break

    end_onnx_inference_batch = time.time()
    # print("Total batch tokenization time (in seconds): ", total_batch_tokenization_time)
    # print("Total inference time (in seconds): ", total_inference_time)
    # print("Total build label time (in seconds): ", total_build_label_time)
    # print("Duration ONNX inference (in seconds) with {} and batch tokenization: ".format(tokenizer_str),
    # print("Duration ONNX inference (in seconds): ",
    #       end_onnx_inference_batch - start_onnx_inference_batch,
    #       (end_onnx_inference_batch - start_onnx_inference_batch) / len(examples))
    # print(onnx_inference)
    return onnx_inference


def get_env_var(varname, default):
    if os.environ.get(varname) != None:
        var = int(os.environ.get(varname))
        # print(varname, ':', var)
    else:
        var = default
        # print(varname, ':', var, '(Default)')
    return var


# # Choose Number of Nodes To Distribute Credentials: e.g. jobarray=0-4, cpu_per_task=20, credentials = 90 (<100)
# SLURM_ARRAY_TASK_ID = get_env_var('SLURM_ARRAY_TASK_ID', 0)
# SLURM_ARRAY_TASK_COUNT = get_env_var('SLURM_ARRAY_TASK_COUNT', 1)
# SLURM_JOB_ID = get_env_var('SLURM_JOB_ID', 1)

SLURM_ARRAY_TASK_ID = 0
SLURM_ARRAY_TASK_COUNT = 1
SLURM_JOB_ID = 0

# print('Hostname:', socket.gethostname())
# print('SLURM_ARRAY_TASK_ID', SLURM_ARRAY_TASK_ID)
# print('SLURM_ARRAY_TASK_COUNT', SLURM_ARRAY_TASK_COUNT)
# ####################################################################################################################################
# # loading data
# ####################################################################################################################################

path_to_data = args.input_path

# print('Load random Tweets:')

start_time = time.time()
#
# paths_to_random = list(np.array_split(
#     glob(os.path.join(path_to_data, '*.parquet')),
#     SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID])
# # print('#files:', len(paths_to_random))
#
# tweets_random = pd.DataFrame()
# for file in paths_to_random:
#     # print(file)
#     tweets_random = pd.concat([tweets_random, pd.read_parquet(file)[['tweet_id', 'text']]])
#     # print(tweets_random.shape)
#
#
# tweets_random = tweets_random.head(NUM_TWEETS)
#
# # print('load random sample:', str(time.time() - start_time), 'seconds')
# # print(tweets_random.shape)
#
# # print('dropping duplicates:')
# # random contains 7.3G of data!!
# start_time = time.time()
# tweets_random = tweets_random.drop_duplicates('text')
# # print('drop duplicates:', str(time.time() - start_time), 'seconds')
# print(tweets_random.shape)

tweets_random = pd.read_csv('/Users/dval/work_temp/twitter_from_nyu/inference/DeepPavlov-bert-base-cased-conversational_mar1_iter4_3297484_seed-10/best_model/job_search.csv')

start_time = time.time()
# print('converting to list')
examples = tweets_random.text.values.tolist()

# print('convert to list:', str(time.time() - start_time), 'seconds')

for column in ["is_unemployed", "lost_job_1mo", "job_search", "is_hired_1mo", "job_offer"]:
    # print('\n\n!!!!!', column)
    loop_start = time.time()
    # best_model_folder = best_model_folders_dict[args.country_code][f'iter{str(args.iteration_number)}'][column]
    # model_path = os.path.join('/scratch/mt4493/twitter_labor/trained_models', args.country_code, best_model_folder,
    # column, 'models', 'best_model')
    # model_path = '/Users/dval/misc/twitter_greene/best_model'
    model_path = '/Users/dval/work_temp/twitter_from_nyu/inference/DeepPavlov-bert-base-cased-conversational_mar1_iter4_3297484_seed-10/best_model/'

    # print(model_path)
    # onnx_path = os.path.join(model_path, 'onnx')
    # print(onnx_path)

    path_best_model = model_path

    train_args = read_json(filename=os.path.join(path_best_model, 'model_args.json'))
    # train_args['eval_batch_size'] = BATCH_SIZE
    train_args['use_cuda'] = False
    best_model = ClassificationModel('bert', path_best_model, args=train_args, use_cuda=False)


    ####################################################################################################################################
    # TOKENIZATION and INFERENCE
    ####################################################################################################################################
    # print('Predictions of random Tweets:')
    start_time = time.time()
    # onnx_labels = inference(os.path.join(onnx_path, 'converted-optimized.onnx'),
    # onnx_labels = inference(os.path.join(onnx_path, 'converted-optimized.onnx'),
    # onnx_labels = inference(os.path.join(onnx_path, 'converted.onnx'),
    # onnx_labels = inference(os.path.join(onnx_path, 'converted-optimized-quantized.onnx'),
    onnx_labels = inference(best_model,
                            examples)

    total_time = float(str(time.time() - start_time))
    per_tweet = total_time / tweets_random.shape[0]
    print('time taken:', total_time, 'seconds')
    print('per tweet:', per_tweet, 'seconds')

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



    break

# Open a file with access mode 'a'
file_object = open('profiling_stats.csv', 'a')
# Append 'hello' at the end of file
file_object.write( str(BATCH_SIZE)+','+str(total_time)+','+str(per_tweet)+'\n')
# Close the file
file_object.close()
