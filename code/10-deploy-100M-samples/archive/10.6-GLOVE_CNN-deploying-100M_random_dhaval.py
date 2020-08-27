import os
import time
import datetime

import tensorflow as tf
from tensorflow.contrib import learn

import numpy as np
import glove_text_cnn.data_utils as utils

from glove_text_cnn.text_cnn import TextCNN
from glove_text_cnn.data_utils import IMDBDataset

import argparse
import pandas as pd
import pickle
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons


root_path='/scratch/da2734/twitter/jobs/running_on_200Msamples/'


def get_env_var(varname, default):
    if os.environ.get(varname) != None:
        var = int(os.environ.get(varname))
        print(varname, ':', var)
    else:
        var = default
        print(varname, ':', var, '(Default)')
    return var

def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()

    # necessary
    # parser.add_argument("--checkpoint_dir", type=str, help="Path to the models", default="/home/manuto/Documents/world_bank/bert_twitter_labor/code/glove-text-cnn/runs/default_run_name/checkpoints")
    # parser.add_argument("--eval_data_path", type=str, help="Path to the evaluation data. Must be in csv format.", default="/home/manuto/Documents/world_bank/bert_twitter_labor/code/twitter/data/may20_9Klabels/data_binary_pos_neg_balanced")
    parser.add_argument("--vocab_path", type=str, help="Path pickle file.", default="/scratch/da2734/twitter/jobs/running_on_200Msamples/iteration1/trained_glove_models_iter1/vocab.pckl")
    parser.add_argument("--label", default='is_unemployed', type=str)
    parser.add_argument("--preprocessing", default=False, type=bool)

    args = parser.parse_args()
    return args

def prepare_filepath_for_storing_pred(eval_data_path: str) -> str:
    path_to_store_pred = os.path.join(os.path.dirname(eval_data_path), 'glove_predictions')
    if not os.path.exists(path_to_store_pred):
        os.makedirs(path_to_store_pred)
    return path_to_store_pred

def tokenizer(text):
    return [wdict.get(w.lower(), 0) for w in text.split(' ')]

def pad_dataset(dataset, maxlen):
    return np.array(
        [np.pad(r, (0, maxlen - len(r)), mode='constant') if len(r) < maxlen else np.array(r[:maxlen])
         for r in dataset])

def create_label(label):
    if label == 1:
        return [0, 1]
    elif label == 0:
        return [1, 0]

args = get_args_from_command_line()
print(args)

print ("Intialising test parameters ...")

batch_size = 64
# Checkpoint directory from training run
# checkpoint_dir = args.checkpoint_dir
# Evaluate on all training data
eval_train = False

# Misc Parameters
allow_soft_placement = True
log_device_placement = False

# Choose Number of Nodes To Distribute Credentials: e.g. jobarray=0-4, cpu_per_task=20, credentials = 90 (<100)
# SLURM_JOB_ID = get_env_var('SLURM_JOB_ID', 0)
SLURM_ARRAY_TASK_ID = get_env_var('SLURM_ARRAY_TASK_ID', 0)
SLURM_ARRAY_TASK_COUNT = get_env_var('SLURM_ARRAY_TASK_COUNT', 1)

# SLURM_JOB_ID = 123123123 #debug
# SLURM_ARRAY_TASK_ID = 10 #debug
# SLURM_ARRAY_TASK_COUNT = 500 #debug


# print('SLURM_JOB_ID', SLURM_JOB_ID)
print('SLURM_ARRAY_TASK_ID', SLURM_ARRAY_TASK_ID)
print('SLURM_ARRAY_TASK_COUNT', SLURM_ARRAY_TASK_COUNT)


# ####################################################################################################################################
# # loading data
# ####################################################################################################################################

import time
import pyarrow.parquet as pq
from glob import glob
import os
import numpy as np

path_to_data='/scratch/spf248/twitter/data/classification/US/'

all_start = time.time()

print('Load random Tweets:')
# random contains 7.3G of data!!
start_time = time.time()

paths_to_random=list(np.array_split(
                        glob(os.path.join(path_to_data,'random_first_half','*.parquet')),
#                         glob(os.path.join(path_to_data,'random_10perct_sample','*.parquet')),
#                         glob(os.path.join(path_to_data,'random_1perct_sample','*.parquet')),
                        SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID])
print('#files:', len(paths_to_random))

tweets_random=pd.DataFrame()
for file in paths_to_random:
    print(file)
    tweets_random=pd.concat([tweets_random,pd.read_parquet(file)[['tweet_id','text']]])       
    print(tweets_random.shape)
    
#     break #DEBUG


print('dropping duplicates:')
# random contains 7.3G of data!!
start_time = time.time()
tweets_random = tweets_random.drop_duplicates('text')
print('drop duplicates:', str(time.time() - start_time), 'seconds')
print(tweets_random.shape)

# tweets_random = tweets_random[:100] #DEBUG

eval_df = tweets_random

print(eval_df.head())

# print ("Loading test data ...")
# eval_df = pd.read_csv(args.eval_data_path, lineterminator='\n')

#Preprocessing
text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
               'time', 'url', 'date', 'number'],
    # terms that will be annotated
    annotate={"hashtag", "allcaps", "elongated", "repeated",
              'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens

    # corpus from which the word statistics are going to be used
    # for word segmentation
    segmenter="twitter",

    # corpus from which the word statistics are going to be used
    # for spell correction
    corrector="twitter",

    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words

    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,

    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)

print('created text_processor', text_processor)

def ekphrasis_preprocessing(tweet):
    return " ".join(text_processor.pre_process_doc(tweet))

if args.preprocessing:
    eval_df['text'] = eval_df['text'].apply(ekphrasis_preprocessing)
    print("*********Text has been preprocessed*********")
with open(args.vocab_path, 'rb') as dfile:
    wdict = pickle.load(dfile)

print('Predictions of random Tweets:')
start_time = time.time()    
# eval_df = eval_df[eval_df['text'].apply(lambda x: isinstance(x, str))]
eval_df['text_tokenized'] = eval_df['text'].apply(tokenizer)
x_test = pad_dataset(eval_df.text_tokenized.values.tolist(), 128)
print('>>>> x_test', type(x_test), x_test.shape)
print('tokenizer+padding:', str(time.time() - start_time), 'seconds')
print('per tweet:', (time.time() - start_time)/x_test.shape[0], 'seconds')

# #y_test = np.array((eval_df['class'].apply(create_label)).values.tolist())

# # Evaluation
for column in ["is_unemployed", "lost_job_1mo", "job_search", "is_hired_1mo", "job_offer"]:
    # checkpoint_file = tf.train.latest_checkpoint('/scratch/da2734/twitter/jobs/running_on_200Msamples/glove_models_june23/glove_cnn_jun3_10Klabels_unbalanced_preprocessing_is_unemployed/checkpoints')
    print('\n\n!!!!!', column)
#     loop_start = time.time()
    
    print('Predictions of random Tweets:')
    start_time = time.time()
    
    checkpoint_file = tf.train.latest_checkpoint('/scratch/da2734/twitter/jobs/running_on_200Msamples/iteration1/trained_glove_models_iter1/glove_cnn_jun27_iter1_new_train_test_split_preprocessing_{}/checkpoints'.format(column))

    # checkpoint_file = tf.train.latest_checkpoint('/scratch/da2734/twitter/jobs/running_on_200Msamples/glove_models_june23/glove_cnn_jun3_10Klabels_unbalanced_preprocessing_{}/checkpoints'.format(column))
    print('> checkpoint_file', checkpoint_file)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=allow_soft_placement,
          log_device_placement=log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            predictions_proba = graph.get_operation_by_name("output/predictions_proba").outputs[0]
            #predictions_proba = predictions_proba[:, 1]
            # Generate batches for one epoch
            batches = utils.batch_iter(list(x_test), batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []
            all_predictions_proba = []
            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                batch_predictions_proba = sess.run(predictions_proba, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])
                all_predictions_proba = np.concatenate([all_predictions_proba, batch_predictions_proba[:, 1]])

    print(column, len(all_predictions_proba))
    
    print('time taken:', str(time.time() - start_time), 'seconds')
    print('per tweet:', (time.time() - start_time)/x_test.shape[0], 'seconds')

    if not os.path.exists(os.path.join(root_path,'iteration1/glove_inferences/glove_inferences_dhaval', column)):
        print('>>>> directory doesnt exists, creating it')
        os.makedirs(os.path.join(root_path,'iteration1/glove_inferences/glove_inferences_dhaval', column))   
    

    print('Save Predictions of random Tweets:')
    start_time = time.time()  
    
    predictions_random_df = pd.DataFrame(data=all_predictions_proba, columns = ['proba'])
    predictions_random_df = predictions_random_df.set_index(tweets_random.tweet_id)

    print(predictions_random_df.head())
    predictions_random_df.to_csv(
        os.path.join(root_path,'iteration1/glove_inferences/glove_inferences_dhaval', column, 'random'+'-'+str(SLURM_ARRAY_TASK_ID)+'.csv')
        # os.path.join(root_path,'iteration1/glove_inferences/glove_inferences_dhaval', column, 'random'+'-'+str(SLURM_JOB_ID)+'-'+str(SLURM_ARRAY_TASK_ID)+'.csv')
        )
    
    print('saved to:\n', os.path.join(root_path,'iteration1/glove_inferences/glove_inferences_dhaval', column, 'random'+'-'+str(SLURM_ARRAY_TASK_ID)+'.csv'), 'saved')
    # print('saved to:\n', os.path.join(root_path,'iteration1/glove_inferences/glove_inferences_dhaval', column, 'random'+'-'+str(SLURM_JOB_ID)+'-'+str(SLURM_ARRAY_TASK_ID)+'.csv'), 'saved')

    print('save time taken:', str(time.time() - start_time), 'seconds')

    print('all_start:', str(time.time() - all_start), ',', (time.time() - all_start)/x_test.shape[0], ',', x_test.shape[0])

    
#     break    
    
    
    
    
    
    
    
    
    
    