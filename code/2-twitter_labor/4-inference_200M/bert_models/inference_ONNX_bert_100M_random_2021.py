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

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info('libs loaded')


parser = argparse.ArgumentParser()

parser.add_argument("--input_path", type=str, help="path to input data")
parser.add_argument("--output_path", type=str, help="path where inference csv is saved")
parser.add_argument("--country_code", type=str, help="path where inference csv is saved")
parser.add_argument("--iteration_number", type=int)


args = parser.parse_args()

logger.info(args)


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


def inference(onnx_model, model_dir, examples, fast_tokenizer, num_threads):
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
    logger.info(onnx_model)
    ort_session = ort.InferenceSession(onnx_model, options)

    # pytorch pretrained model and tokenizer
    if 'bertweet' in onnx_model:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, normalization=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

    tokenizer_str = "TokenizerFast"

    logger.info("**************** {} ONNX inference with batch tokenization and with {} tokenizer****************".format(
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
            logger.info('[inference... ]', i, 'out of ', len(examples))

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
    #         logger.info(i, label[0], onnx_logits.detach().cpu().numpy()[0].tolist(), type(onnx_logits.detach().cpu().numpy()[0]) )

    end_onnx_inference_batch = time.time()
    logger.info("Total batch tokenization time (in seconds): ", total_batch_tokenization_time)
    logger.info("Total inference time (in seconds): ", total_inference_time)
    logger.info("Total build label time (in seconds): ", total_build_label_time)
    logger.info("Duration ONNX inference (in seconds) with {} and batch tokenization: ".format(tokenizer_str),
          end_onnx_inference_batch - start_onnx_inference_batch,
          (end_onnx_inference_batch - start_onnx_inference_batch) / len(examples))

    return onnx_inference


def get_env_var(varname, default):
    if os.environ.get(varname) != None:
        var = int(os.environ.get(varname))
        logger.info(varname, ':', var)
    else:
        var = default
        logger.info(varname, ':', var, '(Default)')
    return var


# Choose Number of Nodes To Distribute Credentials: e.g. jobarray=0-4, cpu_per_task=20, credentials = 90 (<100)
SLURM_ARRAY_TASK_ID = get_env_var('SLURM_ARRAY_TASK_ID', 0)
SLURM_ARRAY_TASK_COUNT = get_env_var('SLURM_ARRAY_TASK_COUNT', 1)
SLURM_JOB_ID = get_env_var('SLURM_JOB_ID', 1)

logger.info('SLURM_ARRAY_TASK_ID', SLURM_ARRAY_TASK_ID)
logger.info('SLURM_ARRAY_TASK_COUNT', SLURM_ARRAY_TASK_COUNT)
# ####################################################################################################################################
# # loading data
# ####################################################################################################################################

path_to_data = args.input_path

logger.info('Load random Tweets:')

start_time = time.time()

paths_to_random = list(np.array_split(
    glob(os.path.join(path_to_data, '*.parquet')),
    SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID])
logger.info('#files:', len(paths_to_random))

tweets_random = pd.DataFrame()
for file in paths_to_random:
    logger.info(file)
    tweets_random = pd.concat([tweets_random, pd.read_parquet(file)[['tweet_id', 'text']]])
    logger.info(tweets_random.shape)

logger.info('load random sample:', str(time.time() - start_time), 'seconds')
logger.info(tweets_random.shape)

logger.info('dropping duplicates:')
# random contains 7.3G of data!!
start_time = time.time()
tweets_random = tweets_random.drop_duplicates('text')
logger.info('drop duplicates:', str(time.time() - start_time), 'seconds')
logger.info(tweets_random.shape)

start_time = time.time()
logger.info('converting to list')
examples = tweets_random.text.values.tolist()

logger.info('convert to list:', str(time.time() - start_time), 'seconds')

best_model_folders_dict = {'iter0': { 'US': {
    'lost_job_1mo': 'vinai_bertweet-base_jan5_iter0_928517_SEED_7',
    'is_hired_1mo': 'vinai_bertweet-base_jan5_iter0_928525_SEED_15',
    'is_unemployed': 'vinai_bertweet-base_jan5_iter0_928513_SEED_3',
    'job_offer': 'roberta-base_jan5_iter0_928467_SEED_2',
    'job_search': 'vinai_bertweet-base_jan5_iter0_928513_SEED_3'
}}}

for column in ["is_unemployed", "lost_job_1mo", "job_search", "is_hired_1mo", "job_offer"]:

    logger.info('\n\n!!!!!', column)
    loop_start = time.time()
    best_model_folder = best_model_folders_dict[f'iter{str(args.iteration_number)}'][args.country_code][column]
    model_path = os.path.join('/scratch/mt4493/twitter_labor/trained_models', args.country_code, best_model_folder, column, 'models', 'best_model')

    logger.info(model_path)
    onnx_path = os.path.join(model_path, 'onnx')
    logger.info(onnx_path)

    ####################################################################################################################################
    # TOKENIZATION and INFERENCE
    ####################################################################################################################################
    logger.info('Predictions of random Tweets:')
    start_time = time.time()
    onnx_labels = inference(os.path.join(onnx_path, 'converted-optimized-quantized.onnx'),
                            model_path,
                            examples,
                            fast_tokenizer=True,
                            num_threads=5)

    logger.info('time taken:', str(time.time() - start_time), 'seconds')
    logger.info('per tweet:', (time.time() - start_time) / tweets_random.shape[0], 'seconds')

    ####################################################################################################################################
    # SAVING
    ####################################################################################################################################
    logger.info('Save Predictions of random Tweets:')
    start_time = time.time()
    final_output_path = args.output_path
    if not os.path.exists(os.path.join(final_output_path, column)):
        logger.info('>>>> directory doesnt exists, creating it')
        os.makedirs(os.path.join(final_output_path, column))
    # create dataframe containing tweet id and probabilities
    predictions_random_df = pd.DataFrame(data=onnx_labels, columns=['first', 'second'])
    predictions_random_df = predictions_random_df.set_index(tweets_random.tweet_id)
    # reformat dataframe
    predictions_random_df = predictions_random_df[['second']]
    predictions_random_df.columns = ['score']

    logger.info(predictions_random_df.head())
    predictions_random_df.to_parquet(
        os.path.join(final_output_path, column,
                     str(getpass.getuser()) + '_random' + '-' + str(SLURM_ARRAY_TASK_ID) + '.parquet'))

    logger.info('saved to:\n', os.path.join(final_output_path, column,
                                      str(getpass.getuser()) + '_random' + '-' + str(SLURM_ARRAY_TASK_ID) + '.parquet'),
          'saved')

    logger.info('save time taken:', str(time.time() - start_time), 'seconds')

    logger.info('full loop:', str(time.time() - loop_start), 'seconds', (time.time() - loop_start) / len(examples))
