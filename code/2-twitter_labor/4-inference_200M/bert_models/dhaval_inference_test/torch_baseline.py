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

# path_to_data = '/scratch/mt4493/twitter_labor/twitter-labor-data/data/random_samples/random_samples_splitted/US/test'
# path_to_data = '/Users/dval/work_temp/twitter_from_nyu/data/random'

# print('Load random Tweets:')

start_time = time.time()

tweets_random = pd.DataFrame()
# print('path_to_data', path_to_data)
# for file in os.listdir(path_to_data):
#     print('reading', file)
#     tweets_random = pd.concat([tweets_random,
#                                pd.read_parquet(path_to_data+'/'+file)[['tweet_id', 'text']]])
#     break
tweets_random = pd.read_parquet('/scratch/mt4493/twitter_labor/twitter-labor-data/data/random_samples/random_samples_splitted/US/test/part-02998-2eecee1d-0c7f-44e8-af29-0810926e4b56-c000.snappy.parquet')[['tweet_id', 'text']]
# tweets_random = pd.read_parquet('/Users/dval/work_temp/twitter_from_nyu/data/random/part-02998-2eecee1d-0c7f-44e8-af29-0810926e4b56-c000.snappy.parquet')[['tweet_id', 'text']]



print('input shape', tweets_random.shape)
print(tweets_random.head())

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



####################################################################################################################################
# TORCH TOKENIZATION and INFERENCE
####################################################################################################################################
torch_model_path = '/scratch/mt4493/twitter_labor/trained_models/US/DeepPavlov-bert-base-cased-conversational_mar1_iter4_3297484_seed-10/job_search/models/best_model/'
# torch_model_path = '/Users/dval/work_temp/twitter_from_nyu/inference/DeepPavlov-bert-base-cased-conversational_mar1_iter4_3297484_seed-10/best_model/'
torch_path_best_model = torch_model_path

train_args = read_json(filename=os.path.join(torch_path_best_model, 'model_args.json'))
# train_args['eval_batch_size'] = BATCH_SIZE
train_args['use_cuda'] = False
torch_best_model = ClassificationModel('bert', torch_path_best_model, args=train_args, use_cuda=False)

for REPLICATION in range(5):

    start_time = time.time()
    torch_labels = torch_inference(torch_best_model,
                            examples)
    torch_total_time = float(str(time.time() - start_time))
    torch_per_tweet = torch_total_time / tweets_random.shape[0]
    # print('time taken:', torch_total_time, 'seconds')
    # print('per tweet:', torch_per_tweet, 'seconds')
    # print(torch_labels)

    torch_predictions_random_df = pd.DataFrame(data=torch_labels, columns=['torch_score'])
    torch_predictions_random_df = torch_predictions_random_df.set_index(tweets_random.tweet_id)
    torch_predictions_random_df['tweet_id'] = torch_predictions_random_df.index
    torch_predictions_random_df = torch_predictions_random_df.reset_index(drop=True)
    torch_predictions_random_df['torch_time_per_tweet'] = torch_per_tweet
    # torch_predictions_random_df['num_tweets'] = NUM_TWEETS
    # torch_predictions_random_df['onnx_batchsize'] = BATCH_SIZE
    # torch_predictions_random_df.columns = ['tweet_id', 'onnx_score']
    # reformat dataframe
    # torch_predictions_random_df = torch_predictions_random_df[['tweet_id', 'second']]
    # print(tweets_random.head() )
    # print(torch_predictions_random_df.head())



    MODEL_TYPE  = 'torch_nyu'

    # final_output_path = '/Users/dval/work_temp/twitter_from_nyu/nyu_temp_output_speedtest_standalone'
    # final_output_path = '/scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/4-inference_200M/bert_models/temp_output_speedtest_standalone'
    final_output_path = '/scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/4-inference_200M/bert_models/dhaval_inference_test/replication_output_data'


    torch_predictions_random_df.to_csv(
    # merged.to_csv(
                os.path.join(final_output_path, 'torch_reference_nyu_rep-{}_nyu.csv'.format(REPLICATION)))

    print('saved to:\n', os.path.join(final_output_path, 'torch_reference_nyu_rep-{}.csv'.format(REPLICATION)))





