# Inference speed up

## ONNX

Here is an example of how to do inference with ONNX. You should log in on the manu-tonneau-twitter-labor-2 machine to perform it.

### Convert PyTorch (simpletransformers) model to ONNX:

- Create an output folder for your ONNX model:
`mkdir test_dhaval_onnx`
- Run the following command to convert the model. 
  - You need to use the version of convert_graph_to_onnx.py from my [transformers fork](https://github.com/manueltonneau/transformers). This transformers folder on the manu-tonneau-twitter-labor-2 disk is based on the latter.
  - Adapt the `tokenizer` to the model you are using (here, by default, 'DeepPavlov/bert-base-cased-conversational' that we usually ConvBERT).
  - For the `model` argument, I put an example below. Feel free to take any other one. You need to point to the folder containing the bin file and not the bin file itself.
  - The last line is the output path. Make sure to replace X by the name of your machine. 

```shell

python3 transformers/src/transformers/convert_graph_to_onnx.py \ 
--framework pt \
--tokenizer DeepPavlov/bert-base-cased-conversational \
--model results_simpletransformers_bert-base-cased-conversational_jun3_10Klabels_0/DeepPavlov/bert-base-cased-conversational_jun3_10Klabels_is_hired_1mo/models/checkpoint-1071-epoch-7 \
/home/X/test_dhaval_onnx/convbert_jun3_is_hired1mo_7.onnx
```

### Optimize your ONNX model


```
pip install onnxruntime-tools 
python -m onnxruntime_tools.optimizer_cli --input test_dhaval_onnx/convbert_jun3_is_hired1mo_7.onnx --output test_dhaval_onnx/convbert_jun3_is_hired1mo_7.onnx --model_type bert
```

There are other options you can use. For instance: 

'The script also provides a flag --float16 to leverage mixed precision performance gains from newer GPUs. '

### Do inference with ONNX on CPU

Based on the [blog post](https://medium.com/microsoftazure/accelerate-your-nlp-pipelines-using-hugging-face-transformers-and-onnx-runtime-2443578f4333), I wrote a script to do inference with ONNX. Without `onnxruntime-gpu` nor CUDA installed, it will run on CPU by default. 

- Activate the `onnx_env` environment
- Run the following command:

```
python3 twitter/code/10-deploy-100M-samples/onnx_predict.py \ 
--onnx_model_path test_dhaval_onnx/convbert_jun3_is_hired1mo_7.onnx \ 
--text <EXAMPLE_INPUT_TEXT_IN_STR_FORMAT>
``` 

The output is printed and corresponds to the scores of each class (in order negative and positive) outputted by the last classification layer. It needs an extra softmax layer to be converted to probabilities. 

### Do inference on GPU

I have not yet managed to run ONNX models on GPUs. One example of how to do this is [here](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/notebooks/PyTorch_Bert-Squad_OnnxRuntime_GPU.ipynb).
It implies installing specific versions of onnxruntime-gpu, CUDA and CUDNN. 

### Is there a tradeoff between accuracy and inference speed?

Apparently **not** according to the official [ONNX Runtime](https://microsoft.github.io/onnxruntime/about.html) website: "Built-in optimization features trim and consolidate nodes without impacting model accuracy.".

To test this, I ran a few tests (using the convbert_jun3_is_hired1mo_7.onnx) to see if output differs between the PT model and its optimized ONNX version, given the same input. The result is that the output is almost exactly the same everytime. 

- Input: "Today is the first day of my new job, so excited"
	- Output scores PT model: [-3.592478 ,  5.0663743]
	- Output scores ONNX model: [-3.592478  5.066375]]
- Input: "Hello I love you"
	- Output scores PT model: [-0.99988693,  0.71829736]
	- Output scores ONNX model: [-0.99988693  0.7182969 ]
- Input "This is a test"
	- Output scores PT model: [ 1.8496819, -2.2533016]
	- Output scores ONNX model: [ 1.8496821 -2.253301 ]

Not sure whether such small differences will strongly affect model performance. 
