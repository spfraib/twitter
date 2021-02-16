# code to convert saved pytorch models to optimized+quantized onnx models for faster inference

import os
import shutil
from transformers.convert_graph_to_onnx import convert, optimize, quantize, verify
import argparse
import logging
from pathlib import Path

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


parser = argparse.ArgumentParser()

parser.add_argument("--iteration_number", type=int, help="path to pytorch models (with onnx model in model_path/onnx/")
parser.add_argument("--country_code", type=str)

args = parser.parse_args()

best_model_paths_dict = {
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
            'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_feb9_iter1_2435288_seed-15',
            'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_feb9_iter1_2435284_seed-11',
            'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_feb9_iter1_2435275_seed-2',
            'job_offer': 'DeepPavlov-bert-base-cased-conversational_feb9_iter1_2435285_seed-12',
            'job_search': 'DeepPavlov-bert-base-cased-conversational_feb9_iter1_2435282_seed-9'},
        'iter2': {
            'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_feb13_iter2_2716731_seed-11',
            'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_feb13_iter2_2716721_seed-1',
            'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_feb13_iter2_2716722_seed-2',
            'job_offer': 'DeepPavlov-bert-base-cased-conversational_feb13_iter2_2716728_seed-8',
            'job_search': 'DeepPavlov-bert-base-cased-conversational_feb13_iter2_2716728_seed-8'
        },
        'iter3': {
            'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_feb16_iter3_2825612_seed-10',
            'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_feb16_iter3_2825613_seed-11',
            'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_feb16_iter3_2825604_seed-2',
            'job_offer': 'DeepPavlov-bert-base-cased-conversational_feb16_iter3_2825614_seed-12',
            'job_search': 'DeepPavlov-bert-base-cased-conversational_feb16_iter3_2825613_seed-11'
        }
    }}

for label in ["lost_job_1mo", "is_unemployed", "job_search", "is_hired_1mo", "job_offer"]:

    logger.info(f'*****************************{label}*****************************')
    model_path = os.path.join('/scratch/mt4493/twitter_labor/trained_models', args.country_code,
                              best_model_paths_dict[args.country_code][f'iter{str(args.iteration_number)}'][label],
                              label, 'models', 'best_model')
    onnx_path = os.path.join(model_path, 'onnx')

    try:
        shutil.rmtree(onnx_path)  # deleting onxx folder and contents, if exists, conversion excepts
    except:
        logger.info('no existing folder, creating one')
        os.makedirs(onnx_path)

    logger.info('>> converting..')
    convert(framework="pt",
            model=model_path,
            tokenizer=convert_model_path_to_model_name(model_path),
            output=Path(os.path.join(onnx_path, 'converted.onnx')),
            opset=11,
            pipeline_name='sentiment-analysis')

    logger.info('>> ONNX optimization')
    optimized_output = optimize(Path(os.path.join(onnx_path, 'converted.onnx')))
    logger.info('>> Quantization')
    quantized_output = quantize(optimized_output)

    logger.info('>> Verification')
    verify(Path(os.path.join(onnx_path, 'converted.onnx')))
    verify(optimized_output)
    verify(quantized_output)
