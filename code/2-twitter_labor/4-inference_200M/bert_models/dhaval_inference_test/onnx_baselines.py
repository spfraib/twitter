import os
import json
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
from simpletransformers.classification import ClassificationModel
from scipy.special import softmax
import scipy
from sklearn.metrics import mean_squared_error

# BATCH_SIZE = int(args.batchsize)

# MODEL_TYPE = 'converted.onnx'
# MODEL_TYPE = 'converted-optimized.onnx'
# MODEL_TYPE = 'converted-optimized-quantized.onnx'

####################################################################################################################################
# HELPER FUNCTIONS
###################################################################################################################################

def read_json(filename: str):
    with open(filename) as f_in:
        return json.load(f_in)

def get_tokens(tokens_dict, i):
    i_tokens_dict = dict()
    for key in ['input_ids', 'token_type_ids', 'attention_mask']:
        i_tokens_dict[key] = tokens_dict[key][i]
    tokens = {name: np.atleast_2d(value) for name, value in i_tokens_dict.items()}
    return tokens


# Splitting a list into N parts of approximately equal length
def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out


def torch_inference(model, examples):
    start_inference = time.time()
    # ort_outs = ort_session.run(None, tokens)
    # ort_outs = ort_session.run(None, minibatch)
    predictions, raw_outputs = model.predict( examples )
    # predictions, raw_outputs = model.predict( [example] )
    scores = np.array([softmax(element)[1] for element in raw_outputs])

    return scores

def onnx_inference(onnx_model, model_dir, examples, NUM_BATCHES):
    quantized_str = ''
    if 'quantized' in onnx_model:
        quantized_str = 'quantized'
    onnx_inference = []
    #     pytorch_inference = []
    # onnx session
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 16 # does not seem to make a difference, always parallelized
    # options.inter_op_num_threads = multiprocessing.cpu_count()
    print(ort.get_device())

    # print(onnx_model)
    ort_session = ort.InferenceSession(onnx_model, options)

    # pytorch pretrained model and tokenizer
    if 'bertweet' in onnx_model:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, normalization=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

    tokenizer_str = "TokenizerFast"

    # print("**************** {} ONNX inference with batch tokenization and with {} tokenizer****************".format(
    #     quantized_str, tokenizer_str))
    start_batch_tokenization = time.time()
    tokens_dict = tokenizer.batch_encode_plus(examples, max_length=128)
    token0 = get_tokens(tokens_dict, 0)

    examples_chunks_list = chunkIt(examples, NUM_BATCHES)
    tokens_dict_list = [tokenizer.batch_encode_plus(chunk, padding='longest') for chunk in examples_chunks_list]
    # tokens_dict_list = [tokenizer.batch_encode_plus(chunk, max_length=128) for chunk in examples_chunks_list]

    minibatches_list = []
    for i, token_batch in enumerate(tokens_dict_list):
        minibatch = {}
        number_examples_in_this_batch = len(token_batch['input_ids'])
        minibatch['input_ids'] = np.stack((
                                    [get_tokens(token_batch, i)['input_ids'][0] for i in range(number_examples_in_this_batch)]
                                    ), axis=0)
        minibatch['token_type_ids'] = np.stack((
                                    [get_tokens(token_batch, i)['token_type_ids'][0] for i in range(number_examples_in_this_batch)]
                                    ), axis=0)
        minibatch['attention_mask'] = np.stack((
                                    [get_tokens(token_batch, i)['attention_mask'][0] for i in range(number_examples_in_this_batch)]
                                    ), axis=0)
        minibatches_list.append(minibatch)

    # tokens_dict = tokenizer.batch_encode_plus(examples, padding='longest')
    total_batch_tokenization_time = time.time() - start_batch_tokenization
    total_inference_time = 0
    total_build_label_time = 0
    start_onnx_inference_batch = time.time()

    # for i in range(len(examples)):
    for i, minibatch in enumerate(minibatches_list):
        """
        Onnx inference with batch tokenization
        """

        # if i % 100 == 0:
            # print(i, '/', NUM_BATCHES)

        tokens = get_tokens(tokens_dict, i)
        # inference
        start_inference = time.time()
        # ort_outs = ort_session.run(None, tokens)
        ort_outs = ort_session.run(None, minibatch)
        total_inference_time = total_inference_time + (time.time() - start_inference)
        # build label
        start_build_label = time.time()
        torch_onnx_output = torch.tensor(ort_outs[0], dtype=torch.float32)
        onnx_logits = F.softmax(torch_onnx_output, dim=1)
        logits_label = torch.argmax(onnx_logits, dim=1)
        label = logits_label.detach().cpu().numpy()

        # TODO might be able to make this faster by using arrays with pre-defined size isntead of mutating lists like this
        onnx_inference = onnx_inference + onnx_logits.detach().cpu().numpy().tolist()

    end_onnx_inference_batch = time.time()

    return onnx_inference


def get_env_var(varname, default):
    if os.environ.get(varname) != None:
        var = int(os.environ.get(varname))
        # print(varname, ':', var)
    else:
        var = default
        # print(varname, ':', var, '(Default)')
    return var


# ####################################################################################################################################
# # loading data
# ####################################################################################################################################

path_to_data = '/scratch/mt4493/twitter_labor/twitter-labor-data/data/random_samples/random_samples_splitted/US/test'
# path_to_data = '/Users/dval/work_temp/twitter_from_nyu/data/random'

# print('Load random Tweets:')

start_time = time.time()

tweets_random = pd.DataFrame()
# print('path_to_data', path_to_data)
for file in os.listdir(path_to_data):
    print('reading', file)
    tweets_random = pd.concat([tweets_random,
                               pd.read_parquet(path_to_data+'/'+file)[['tweet_id', 'text']]])
    break

# print('input shape', tweets_random.shape)
NUM_TWEETS = 1000
tweets_random = tweets_random.head(NUM_TWEETS)

tweets_random = tweets_random.drop_duplicates('text')

start_time = time.time()
# print('converting to list')
examples = tweets_random.text.values.tolist()

# print('convert to list:', str(time.time() - start_time), 'seconds')

# for column in ["is_unemployed", "lost_job_1mo", "job_search", "is_hired_1mo", "job_offer"]:
    # loop_start = time.time()

column = "job_search"

onnx_model_path = '/scratch/mt4493/twitter_labor/trained_models/US/DeepPavlov-bert-base-cased-conversational_mar1_iter4_3297484_seed-10/job_search/models/best_model/'
# onnx_model_path = '/Users/dval/work_temp/twitter_from_nyu/inference/DeepPavlov-bert-base-cased-conversational_mar1_iter4_3297484_seed-10/best_model'


onnx_path = os.path.join(onnx_model_path, 'onnx')

for model_type in ['converted.onnx', 'converted-optimized.onnx', 'converted-optimized-quantized.onnx']:
    for batchsize in [1, 2, 5, 10, 20]:
    # for batchsize in [1, 2, 5, 10, 20, 50, 100]:
        for replication in range(5):

            BATCH_SIZE = batchsize
            # BATCH_SIZE = 1
            print(model_type, batchsize, replication)
            NUM_BATCHES = int(np.ceil( NUM_TWEETS/BATCH_SIZE ))
            MODEL_TYPE = model_type
            REPLICATION = replication

            ####################################################################################################################################
            # ONNX TOKENIZATION and INFERENCE
            ####################################################################################################################################
            # print('Predictions of random Tweets:')
            start_time = time.time()
            onnx_labels = onnx_inference(os.path.join(onnx_path, MODEL_TYPE),
                                    onnx_model_path,
                                    examples, 
                                    NUM_BATCHES)

            onnx_total_time = float(str(time.time() - start_time))
            onnx_per_tweet = onnx_total_time / tweets_random.shape[0]


            # ####################################################################################################################################
            # # CALCULATIONS
            # ####################################################################################################################################
            # print('Save Predictions of random Tweets:')
            # start_time = time.time()
            final_output_path = '/scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/4-inference_200M/bert_models/dhaval_inference_test/replication_output_data'
            # final_output_path = '/Users/dval/work_temp/twitter_from_nyu/output'
            if not os.path.exists(os.path.join(final_output_path)):
                # print('>>>> directory doesnt exists, creating it')
                os.makedirs(os.path.join(final_output_path))
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
                        os.path.join(final_output_path,
                         str(getpass.getuser()) + '_random' + '-' +
                                           str(MODEL_TYPE) + '-' +
                                           'bs-' + str(BATCH_SIZE) + '-' +
                                           'rep-' + str(REPLICATION) + '-' +
                                             '.csv'))

            print('saved to:\n', os.path.join(final_output_path,
                  str(getpass.getuser()) + '_random' + '-' +
                                              str(MODEL_TYPE) + '-' +
                                              'bs-' + str(BATCH_SIZE) + '-' +
                                              'rep-' + str(REPLICATION) + '-' +
                                              '.csv'))

#             break
#         break
#     break




