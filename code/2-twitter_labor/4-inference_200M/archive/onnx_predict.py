import numpy as np
from transformers import BertTokenizerFast
import onnxruntime
from onnxruntime import ExecutionMode, InferenceSession, SessionOptions
import time
import argparse
from transformers import squad_convert_examples_to_features

def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    # necessary
    parser.add_argument("--pretrained_model", type=str, help="Name of pretrained model on hugging face", default='DeepPavlov/bert-base-cased-conversational')
    parser.add_argument("--onnx_model_path", type=str, help="Path to onnx model")
    parser.add_argument("--text", type=str, help="Text to classify")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args_from_command_line()

    start = time.time()
    tokenizer = BertTokenizerFast.from_pretrained(args.pretrained_model)
    tokens = tokenizer.encode_plus(args.text)
    tokens = {name: np.atleast_2d(value) for name, value in tokens.items()}
    end_tokenizer = time.time()
    print("Tokenization time: ", end_tokenizer - start)

    start_session = time.time()
    options= SessionOptions()
    options.intra_op_num_threads=1
    options.execution_mode = ExecutionMode.ORT_SEQUENTIAL
    #options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    session = InferenceSession(args.onnx_model_path, options)
    end_session = time.time()
    print("Setting up session time: ", end_session - start_session)

    start_predict = time.time()
    output, = session.run(None, tokens)
    end_predict = time.time()
    print("Predict time: ", end_predict - start_predict)

    print("Overall time: ", end_predict - start)
    print(output)
