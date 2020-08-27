# Launching inference with BERT-based models on 100M samples

## Preliminary steps:

- Make sure you have permission to write to Manu's folder.

## Launch inference:

### 1. Convert PyTorch models to ONNX format:
```
$ python onnx_model_conversion.py /scratch/mt4493/twitter_labor/trained_models/<MODEL_FOLDER>/{}/models/best_model
```
where <MODEL_FOLDER> is the path to the folder containing the model file we want to convert to ONNX format.

An example usage is:
```
$ python onnx_model_conversion.py /scratch/mt4493/twitter_labor/trained_models/iter0/jul23_iter0/DeepPavlov_bert-base-cased-conversational_jul23_iter0_preprocessed_11232989/{}/models/best_model
```

Note that the {} is important to specify label of model.

### 2. Run inference
```
$ sbatch --array=0-3000 ONYX_BERT_labels_deploy_100M_random_iteration_2.sbatch <MODEL_FOLDER>
```

where <MODEL_FOLDER> is the path to the folder inside `/scratch/mt4493/twitter_labor/trained_models/` containing the model file we want to use for inference (e.g. `jul23_iter0/DeepPavlov_bert-base-cased-conversational_jul23_iter0_preprocessed_11232989`).

Note: 
-  This file runs the following command: 
```
$ python -u inference_ONNX_bert_100M_random.py --model_path ${MODEL_PATH}/{}/models/best_model --output_path ${OUTPUT_PATH} > /scratch/mt4493/twitter_labor/code/twitter/jobs/inference_200M/inference_output/iteration2/logs/${SLURM_ARRAY_TASK_ID}-$(date +%s).log 2>&1
```

where `MODEL_PATH=/scratch/mt4493/twitter_labor/trained_models/${MODEL_FOLDER}` and `OUTPUT_PATH=/scratch/mt4493/twitter_labor/twitter-labor-data/data/inference`.

- for BERT, you need at least 5cpus for speeds ups. For GloVe, only 1 cpu is needed. 

- To check on progress, run: `watch -n1 squeue -u <NETID>`


## TODO:
1. move all code to Manu's repo (he'll need to set up git (?) and do a pull) so we can all read and write to his repo
2. use above template but for GLOVE 