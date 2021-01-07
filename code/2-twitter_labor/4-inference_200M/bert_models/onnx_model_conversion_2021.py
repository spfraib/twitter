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

parser = argparse.ArgumentParser()

parser.add_argument("--model_path", help="path to pytorch models (with onnx model in model_path/onnx/")
parser.add_argument("--model_name", help="Name of model in HF model hub")

args = parser.parse_args()

for label in ["lost_job_1mo", "is_unemployed", "job_search", "is_hired_1mo", "job_offer"]:

    logger.info(f'*****************************{label}*****************************')
    model_path = args.model_path.format(label)
    onnx_path = model_path + '/onnx'.format(label)

    try:
        shutil.rmtree(onnx_path)  # deleting onxx folder and contents, if exists, conversion excepts
    except:
        logger.info('no existing folder, creating one')
        os.makedirs(onnx_path)

    logger.info('>> converting..')
    convert(framework="pt",
            model=model_path,
            tokenizer=args.model_name,
            output=Path(os.path.join(onnx_path, 'converted.onnx')),
            opset=11,
            pipeline_name='sentiment-analysis')

    logger.info('>> ONNX optimization')
    optimized_output = optimize(Path(os.path.join(onnx_path, 'converted.onnx')))
    logger.info('>> Quantization')
    quantized_output = quantize(optimized_output)

    logger.info('Verification')
    verify(os.path.join(onnx_path, 'converted.onnx'))
    verify(optimized_output)
    verify(quantized_output)
