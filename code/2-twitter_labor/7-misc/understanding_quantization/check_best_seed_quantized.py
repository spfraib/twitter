from transformers.convert_graph_to_onnx import convert, optimize, quantize, verify
from pathlib import Path
import torch
import onnxruntime as ort
import pandas as pd
import numpy as np
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
from scipy.special import softmax
import scipy
from sklearn import metrics
import logging
import time

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_model_path_to_model_name(model_path):
    if 'bertweet' in model_path:
        return 'vinai/bertweet-base'
    elif 'roberta-base' in model_path:
        return 'roberta-base'
    elif 'DeepPavlov' in model_path:
        return 'DeepPavlov/bert-base-cased-conversational'


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
    options.intra_op_num_threads = 16  # does not seem to make a difference, always parallelized
    # options.inter_op_num_threads = multiprocessing.cpu_count()

    # logger.info(onnx_model)
    ort_session = ort.InferenceSession(onnx_model, options)

    # pytorch pretrained model and tokenizer
    if 'bertweet' in onnx_model:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, normalization=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

    tokenizer_str = "TokenizerFast"

    # logger.info("**************** {} ONNX inference with batch tokenization and with {} tokenizer****************".format(
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
        # logger.info('y')
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
        # logger.info(i, '/', NUM_BATCHES)

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
        #         onnx_inference.append(label[0])
        # onnx_inference.append(onnx_logits.detach().cpu().numpy().tolist())

        # TODO might be able to make this faster by using arrays with pre-defined size isntead of mutating lists like this
        onnx_inference = onnx_inference + onnx_logits.detach().cpu().numpy().tolist()
        # onnx_inference.append(onnx_logits.detach().cpu().numpy()[0].tolist())
        # total_build_label_time = total_build_label_time + (time.time() - start_build_label)
    #         logger.info(i, label[0], onnx_logits.detach().cpu().numpy()[0].tolist(), type(onnx_logits.detach().cpu().numpy()[0]) )

    end_onnx_inference_batch = time.time()
    # logger.info("Total batch tokenization time (in seconds): ", total_batch_tokenization_time)
    # logger.info("Total inference time (in seconds): ", total_inference_time)
    # logger.info("Total build label time (in seconds): ", total_build_label_time)
    # logger.info("Duration ONNX inference (in seconds) with {} and batch tokenization: ".format(tokenizer_str),
    # logger.info("Duration ONNX inference (in seconds): ",
    #       end_onnx_inference_batch - start_onnx_inference_batch,
    #       (end_onnx_inference_batch - start_onnx_inference_batch) / len(examples))
    # logger.info(onnx_inference)
    return onnx_inference


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str, default='US')
    args = parser.parse_args()
    return args


def convert_score_to_predictions(score):
    if score > 0.5:
        return 1
    elif score <= 0.5:
        return 0


if __name__ == '__main__':
    args = get_args_from_command_line()
    model_folder_path = f'/scratch/mt4493/twitter_labor/trained_models/{args.country_code}'
    data_folder_dict = {
        'US': ['jan5_iter0', 'feb22_iter1', 'feb23_iter2', 'feb25_iter3', 'mar1_iter4']}
    labels = ['is_hired_1mo', 'lost_job_1mo', 'is_unemployed', 'job_search', 'job_offer']
    best_model_paths_dict = {
        'US': {
            'iter0': {
                'lost_job_1mo': 'DeepPavlov_bert-base-cased-conversational_jan5_iter0_928497_SEED_14',
                'is_hired_1mo': 'DeepPavlov_bert-base-cased-conversational_jan5_iter0_928488_SEED_5',
                'is_unemployed': 'DeepPavlov_bert-base-cased-conversational_jan5_iter0_928498_SEED_15',
                'job_offer': 'DeepPavlov_bert-base-cased-conversational_jan5_iter0_928493_SEED_10',
                'job_search': 'DeepPavlov_bert-base-cased-conversational_jan5_iter0_928486_SEED_3'
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
            }}}
    results_dict = dict()
    best_model_dict = dict()
    for count, data_folder in enumerate(data_folder_dict[args.country_code]):
        relevant_model_folders = [model_folder for model_folder in os.listdir(model_folder_path) if
                                  data_folder in model_folder]
        results_dict[count] = dict()
        best_model_dict[count] = dict()

        logger.info(f'*** Iteration {count} ***')
        for label in labels:
            logger.info(f'** Label: {label} **')
            results_dict[count][label] = dict()
            best_model_dict[count][label] = dict()
            for model_folder in relevant_model_folders:
                folder_path = os.path.join(model_folder_path, model_folder)
                model_path = os.path.join(folder_path,
                                          label, 'models', 'best_model')
                if os.path.exists(model_path) and 'config.json' in os.listdir(model_path):
                    onnx_path = os.path.join(model_path, 'onnx')
                    if not os.path.exists(onnx_path):
                        logger.info('Converting model to ONNX')
                        os.makedirs(onnx_path)

                        convert(framework="pt",
                                model=model_path,
                                tokenizer=convert_model_path_to_model_name(model_path),
                                output=Path(os.path.join(onnx_path, 'converted.onnx')),
                                opset=11,
                                pipeline_name='sentiment-analysis')

                        optimized_output = optimize(Path(os.path.join(onnx_path, 'converted.onnx')))
                        quantized_output = quantize(optimized_output)

                        verify(Path(os.path.join(onnx_path, 'converted.onnx')))
                        verify(optimized_output)
                        verify(quantized_output)

                    # load evaluation data
                    path_data = '/scratch/mt4493/twitter_labor/twitter-labor-data/data/train_test'
                    path_evaluation_data = os.path.join(path_data, args.country_code, data_folder, 'train_test',
                                                        f'val_{label}.csv')
                    val_df = pd.read_csv(path_evaluation_data, lineterminator='\n')
                    val_df = val_df[['tweet_id', 'text', "class"]]
                    val_df.columns = ['tweet_id', 'text', 'labels']
                    examples = val_df['text'].tolist()
                    # run inference
                    NUM_TWEETS = len(examples)
                    BATCH_SIZE = 1
                    NUM_BATCHES = int(np.ceil(NUM_TWEETS / BATCH_SIZE))
                    logger.info('Start inference')
                    start_inference = time.time()
                    onnx_labels = inference(os.path.join(onnx_path,
                                                         'converted-optimized-quantized.onnx'),
                                            model_path,
                                            examples)
                    end_inference = time.time()
                    logger.info(f'Inference lasted {end_inference - start_inference} seconds.')
                    scores = [element[1] for element in onnx_labels]
                    y_pred = np.vectorize(convert_score_to_predictions)(scores)
                    # compute AUC
                    fpr, tpr, thresholds = metrics.roc_curve(val_df['labels'], scores)
                    auc_eval = metrics.auc(fpr, tpr)
                    results_dict[count][label][model_folder] = auc_eval
            best_model_dict[count][label]['best_pytorch'] = best_model_paths_dict[args.country_code][f"iter{count}"][
                label]
            best_model_dict[count][label]['best_quantized'] = max(results_dict[count][label],
                                                                  key=results_dict[count][label].get)
            best_model_dict[count][label]['same_best'] = best_model_paths_dict[args.country_code][f"iter{count}"][
                                                             label] == max(results_dict[count][label],
                                                                           key=results_dict[count][label].get)
            logger.info(best_model_dict[count][label])
    results_list = list()
    results_df = pd.DataFrame.from_dict(best_model_dict)
    for iter_number in range(5):
        results_iter_df = results_df[iter_number].apply(pd.Series).reset_index()
        results_iter_df['iter'] = iter_number
        results_list.append(results_iter_df)
        # logger.info(results_iter_df)
    results_df = pd.concat(results_list).reset_index(drop=True)
    results_df = results_df.sort_values(by=['index', 'iter']).reset_index(drop=True)

    output_path = f'/scratch/mt4493/twitter_labor/twitter-labor-data/data/debugging/check_best_seed'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    results_df.to_csv(os.path.join(output_path, f'{args.country_code}.csv'), index=False)