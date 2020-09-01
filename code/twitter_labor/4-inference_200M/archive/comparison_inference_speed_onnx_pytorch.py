import torch
from transformers import BertConfig, BertTokenizer, BertTokenizerFast, BertForSequenceClassification
import onnxruntime as ort
from onnxruntime_tools import optimizer
import argparse
import pandas as pd
import numpy as np
from transformers.convert_graph_to_onnx import convert
import os
import time
import torch.nn.functional as F
import onnx
from quantize import quantize, QuantizationMode


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    # necessary
    parser.add_argument("--model_dir", type=str, help="Name of pretrained model on hugging face",
                        default='DeepPavlov/bert-base-cased-conversational')
    parser.add_argument("--onnx_model_path", type=str, help="Path to onnx model")
    parser.add_argument("--eval_data_path", type=str, help="Path to eval csv")
    args = parser.parse_args()
    return args


def preprocess(tokenizer, text):
    max_seq_length = 128
    tokens = tokenizer.tokenize(text)
    tokens.insert(0, "[CLS]")
    tokens.append("[SEP]")
    segment_ids = []
    for i in range(len(tokens)):
        segment_ids.append(0)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    input_ids = torch.tensor([input_ids], dtype=torch.long)
    input_mask = torch.tensor([input_mask], dtype=torch.long)
    segment_ids = torch.tensor([segment_ids], dtype=torch.long)

    return input_ids, input_mask, segment_ids


"""
Inference on pretrained pytorch model
"""


def inference_pytorch(model, input_ids, input_mask, segment_ids, quantization, num_threads):
    torch.set_num_threads(num_threads)
    if quantization:
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    with torch.no_grad():
        outputs = model(input_ids, input_mask, segment_ids)

    logits = outputs[0]
    logits = F.softmax(logits, dim=1)
    return logits


def convert_bert_to_onnx(text, model_dir, onnx_model_path):
    config = BertConfig.from_pretrained(model_dir)
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir, config=config)
    model.to("cpu")
    input_ids, input_mask, segment_ids = preprocess(tokenizer, text)

    dynamic_axes = {
        'input_id': {0:'1',1:'128'},
        'sequence_id': {0:'1',1:'128'},
        'input_mask': {0:'1',1:'128'},
        'output': {0:'1'},
    }
    print('starting export')
    torch.onnx.export(model, (input_ids, input_mask, segment_ids), onnx_model_path,
                      input_names=["input_ids", "input_mask", "segment_ids"],
                      output_names=["output"], opset_version=10, do_constant_folding=True, dynamic_axes=dynamic_axes, verbose=False)

    print("SST model convert to onnx format successfully")

def get_tokens(tokens_dict, i):
    i_tokens_dict = dict()
    for key in ['input_ids','token_type_ids','attention_mask']:
        i_tokens_dict[key] = tokens_dict[key][i]
    tokens = {name: np.atleast_2d(value) for name, value in i_tokens_dict.items()}
    return tokens


def inference(onnx_model, model_dir, examples, fast_tokenizer, num_threads):
    quantized_str = ''
    if 'quantized' in onnx_model:
        quantized_str = 'quantized'
    onnx_inference = []
    pytorch_inference = []
    # onnx session
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 1
    ort_session = ort.InferenceSession(onnx_model, options)
    # pytorch pretrained model and tokenizer
    if fast_tokenizer:
        tokenizer = BertTokenizerFast.from_pretrained(model_dir)
        tokenizer_str = "BertTokenizerFast"

    else:
        tokenizer = BertTokenizer.from_pretrained(model_dir)
        tokenizer_str = "BertTokenizer"
    config = BertConfig.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir, config=config)
    #model.to("cpu")
    print("**************** {} ONNX inference with batch tokenization and with {} tokenizer****************".format(quantized_str, tokenizer_str))
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
        tokens = get_tokens(tokens_dict, i)
        #inference
        start_inference = time.time()
        ort_outs = ort_session.run(None, tokens)
        total_inference_time = total_inference_time + (time.time() - start_inference)
        #build label
        start_build_label = time.time()
        torch_onnx_output = torch.tensor(ort_outs[0], dtype=torch.float32)
        onnx_logits = F.softmax(torch_onnx_output, dim=1)
        logits_label = torch.argmax(onnx_logits, dim=1)
        label = logits_label.detach().cpu().numpy()
        onnx_inference.append(label[0])
        total_build_label_time = total_build_label_time + (time.time() - start_build_label)
    end_onnx_inference_batch = time.time()
    print("Total batch tokenization time (in seconds): ", total_batch_tokenization_time)
    print("Total inference time (in seconds): ", total_inference_time)
    print("Total build label time (in seconds): ", total_build_label_time)
    print("Duration ONNX inference (in seconds) with {} and batch tokenization: ".format(tokenizer_str), end_onnx_inference_batch - start_onnx_inference_batch)

    print("****************{} ONNX inference without batch tokenization and with {} tokenizer****************".format(quantized_str, tokenizer_str))
    start_onnx_inference_no_batch = time.time()
    total_tokenization_time = 0
    total_inference_time = 0
    total_build_label_time = 0
    for example in examples:
        """
        Onnx inference without batch tokenization 
        """
        #input_ids, input_mask, segment_ids = preprocess(tokenizer, example)
        #tokenization
        start_tokenization = time.time()
        tokens = tokenizer.encode_plus(example)
        tokens = {name: np.atleast_2d(value) for name, value in tokens.items()}
        total_tokenization_time = total_tokenization_time + (time.time() - start_tokenization)
        #inference
        start_inference = time.time()
        ort_outs = ort_session.run(None, tokens)
        total_inference_time = total_inference_time + (time.time() - start_inference)
        #build_label
        start_build_label = time.time()
        torch_onnx_output= torch.tensor(ort_outs[0], dtype=torch.float32)
        onnx_logits = F.softmax(torch_onnx_output, dim=1)
        logits_label = torch.argmax(onnx_logits, dim=1)
        label = logits_label.detach().cpu().numpy()
        onnx_inference.append(label[0])
        total_build_label_time = total_build_label_time + (time.time() - start_build_label)

    end_onnx_inference_no_batch = time.time()
    print("One-by-one total tokenization time (in seconds): ", total_tokenization_time)
    print("Total inference time (in seconds): ", total_inference_time)
    print("Total build label time (in seconds): ", total_build_label_time)
    print("Duration ONNX inference (in seconds) with {} and one-by-one tokenization: ".format(tokenizer_str), end_onnx_inference_no_batch - start_onnx_inference_no_batch)

    print("****************Torch inference without batch tokenization, without quantization and with {} tokenizer****************".format(tokenizer_str))
    start_torch_inference_no_quantization = time.time()
    total_tokenization_time = 0
    total_inference_time = 0
    total_build_label_time = 0
    for example in examples:
        """
        Pretrained bert pytorch model
        """
        # tokenization
        start_tokenization = time.time()
        input_ids, input_mask, segment_ids = preprocess(tokenizer, example)
        total_tokenization_time = total_tokenization_time + (time.time() - start_tokenization)
        # inference
        start_inference = time.time()
        torch_out = inference_pytorch(model, input_ids, input_mask, segment_ids, quantization=False, num_threads=num_threads)
        total_inference_time = total_inference_time + (time.time() - start_inference)
        # build label
        start_build_label = time.time()
        logits_label = torch.argmax(torch_out, dim=1)
        label = logits_label.detach().cpu().numpy()
        pytorch_inference.append(label[0])
        total_build_label_time = total_build_label_time + (time.time() - start_build_label)

    end_torch_inference_no_quantization = time.time()
    print("One-by-one total tokenization time (in seconds): ", total_tokenization_time)
    print("Total inference time (in seconds): ", total_inference_time)
    print("Total build label time (in seconds): ", total_build_label_time)
    print("Duration PyTorch inference (in seconds) with {}, without quantization and with {} threads: ".format(tokenizer_str, num_threads), end_torch_inference_no_quantization - start_torch_inference_no_quantization )

    print("****************Torch inference without batch tokenization, with quantization and with {} tokenizer****************".format(tokenizer_str))

    start_torch_inference_w_quantization = time.time()
    total_tokenization_time = 0
    total_inference_time = 0
    total_build_label_time = 0
    for example in examples:
        """
        Pretrained bert pytorch model
        """
        # tokenization
        start_tokenization = time.time()
        input_ids, input_mask, segment_ids = preprocess(tokenizer, example)
        total_tokenization_time = total_tokenization_time + (time.time() - start_tokenization)
        # inference
        start_inference = time.time()
        torch_out = inference_pytorch(model, input_ids, input_mask, segment_ids, quantization=False, num_threads=num_threads)
        total_inference_time = total_inference_time + (time.time() - start_inference)
        # build label
        start_build_label = time.time()
        logits_label = torch.argmax(torch_out, dim=1)
        label = logits_label.detach().cpu().numpy()
        pytorch_inference.append(label[0])
        total_build_label_time = total_build_label_time + (time.time() - start_build_label)

    end_torch_inference_w_quantization = time.time()
    print("One-by-one total tokenization time (in seconds): ", total_tokenization_time)
    print("Total inference time (in seconds): ", total_inference_time)
    print("Total build label time (in seconds): ", total_build_label_time)
    print("Duration PyTorch inference (in seconds) with {} and with quantization and with {} threads: ".format(tokenizer_str, num_threads), end_torch_inference_w_quantization - start_torch_inference_w_quantization )

    #
    # # compare ONNX Runtime and PyTorch results
    # np.testing.assert_allclose(to_numpy(torch_out), onnx_logits, rtol=1e-03, atol=1e-05)
    #
    # print("Exported model has been tested with ONNXRuntime, and the result looks good!")
    return onnx_inference, pytorch_inference


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def remove_initializer_from_input(input,output):

    model = onnx.load(input)
    if model.ir_version < 4:
        print(
            'Model with ir_version below 4 requires to include initilizer in graph input'
        )
        return

    inputs = model.graph.input
    name_to_input = {}
    for i in inputs:
        name_to_input[i.name] = i

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    onnx.save(model, output)

if __name__ == '__main__':
    args = get_args_from_command_line()
    #text = "tick tock tick"
    #convert_bert_to_onnx('tick tock', args.model_dir, args.onnx_model_path)
    #remove_initializer_from_input(args.onnx_model_path, args.onnx_model_path)
    convert(framework="pt", model=args.model_dir, tokenizer="DeepPavlov/bert-base-cased-conversational",
            output=args.onnx_model_path, opset=11)

    # ONNX optimization
    optimized_model = optimizer.optimize_model(args.onnx_model_path, model_type='bert', num_heads=12, hidden_size=768)
    optimized_onnx_model_path = os.path.join(os.path.dirname(args.onnx_model_path), 'bert_optimized.onnx')
    optimized_model.save_model_to_file(optimized_onnx_model_path)
    print('Optimized model saved at :', optimized_onnx_model_path)
    # ONNX quantization
    model = onnx.load(optimized_onnx_model_path)
    quantized_model = quantize(model, quantization_mode=QuantizationMode.IntegerOps, static=False)
    optimized_quantized_onnx_model_path = os.path.join(os.path.dirname(optimized_onnx_model_path), 'bert_optimized_quantized.onnx')
    onnx.save(quantized_model, optimized_quantized_onnx_model_path)
    print('Quantized&optimized model saved at :', optimized_quantized_onnx_model_path)

    #load data
    eval_df = pd.read_csv(args.eval_data_path)
    eval_df = eval_df[['text', 'class']]
    print('Number of examples: ', eval_df.shape[0])
    examples = eval_df.text.values.tolist()
    # labels = eval_df.class.values.tolist()

    # launch inference without quantization

    print("\n ************ Inference without quantization ************ \n")

    onnx_labels, pytorch_labels = inference(optimized_onnx_model_path, args.model_dir, examples, fast_tokenizer=True, num_threads=1)
    onnx_labels, pytorch_labels = inference(optimized_onnx_model_path, args.model_dir, examples, fast_tokenizer=False, num_threads=1)

    # launch inference with quantization

    print("\n ************ Inference with quantization ************ \n")

    onnx_labels, pytorch_labels = inference(optimized_quantized_onnx_model_path, args.model_dir, examples, fast_tokenizer=True, num_threads=1)
    onnx_labels, pytorch_labels = inference(optimized_quantized_onnx_model_path, args.model_dir, examples, fast_tokenizer=False, num_threads=1)
    #start_time_w_fast_tokenizer = time.time()
    #onnx_labels, pytorch_labels = inference(optimized_model_path, args.model_dir, examples, fast_tokenizer=True)
    #print("\n ************ \n")

    #print("total time with fast tokenizer ", time.time() - start_time_w_fast_tokenizer)
    # print("accuracy score of pytorch model", accuracy_score(labels[:top_n], pytorch_labels[:top_n]))
    # print("accuracy score of onnx model", accuracy_score(labels[:top_n], onnx_labels[:top_n]))