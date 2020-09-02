# code to convert saved pytorch models to optimized+quantized onnx models for faster inference

import torch
import onnxruntime as ort
import argparse
import pandas as pd
import numpy as np
import os
import time
import torch.nn.functional as F
import onnx
import sys
import shutil
from onnxruntime.quantization import QuantizationMode, quantize

sys.path.append('/scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/4-inference_200M/bert_models/utils_for_inference')
from transformers.convert_graph_to_onnx import convert
from transformers import BertConfig, BertTokenizer, BertTokenizerFast, BertForSequenceClassification
from onnxruntime_tools import optimizer

model_path_from_terminal = sys.argv[1] 
# e.g. '/scratch/mt4493/2-twitter_labor/trained_models/iter0/jul23_iter0/DeepPavlov_bert-base-cased-conversational_jul23_iter0_preprocessed_11232989/{}/models/best_model'

for label in ["lost_job_1mo","is_unemployed", "job_search", "is_hired_1mo", "job_offer"]:

    print('*****************************{}*****************************'.format(label))
    model_path = model_path_from_terminal.format(label)
    onnx_path = model_path+'/onnx/'.format(label)

    try:
        shutil.rmtree(onnx_path) # deleting onxx folder and contents, if exists, conversion excepts
    except:
        print('no existing folder, creating one')
        os.makedirs(onnx_path)
    
    print('>> converting..')
    convert(framework="pt", 
        model=model_path, 
        tokenizer="DeepPavlov/bert-base-cased-conversational",
        output=onnx_path+'converted.onnx', 
        opset=11)

    print('>> optimizing..')
    # ONNX optimization
    optimized_model = optimizer.optimize_model(onnx_path+'/converted.onnx',
                                               model_type='bert', 
                                               num_heads=12, 
                                               hidden_size=768)

    optimized_onnx_model_path = os.path.join(onnx_path, 'bert_optimized.onnx')
    optimized_model.save_model_to_file(optimized_onnx_model_path)
    print('Optimized model saved at :', optimized_onnx_model_path)

    print('>> quantizing..')    
    model = onnx.load(onnx_path+'/converted.onnx')
    quantized_model = quantize(model=model, quantization_mode=QuantizationMode.IntegerOps, force_fusions=True, symmetric_weight=True)
    optimized_quantized_onnx_model_path = os.path.join(os.path.dirname(optimized_onnx_model_path), 'bert_optimized_ONNXquantized.onnx')
    onnx.save_model(quantized_model, optimized_quantized_onnx_model_path)
    print('Quantized&optimized model saved at :', optimized_quantized_onnx_model_path)
    
    # break
