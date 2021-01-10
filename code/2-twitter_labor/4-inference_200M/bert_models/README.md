# Launching inference with BERT-based models on 100M samples

## Preliminary steps:

- Make sure you have permission to write to Manu's folder.

## Launch inference:

### 1. Convert PyTorch models to ONNX format:
```
$ cd /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/4-inference_200M/bert_models
$ sbatch onnx_model_conversion.sbatch <COUNTRY_CODE> <ITER>
```
where `<COUNTRY_CODE>` is in `['US', 'MX', 'BR']` and `<ITER>` refers to the active learning iteration number (starts at 0).

When running the above command, the best models for each class for the specified iteration, which are listed in the dictionary `best_model_paths_dict` in the script `onnx_model_conversion.py`, are converted to ONNX, optimized and quantized.
### 2. Run inference
```
$ sbatch --array=0-900 inference_ONNX_bert_100M_random.sbatch <COUNTRY_CODE> <ITER> <MODE> <RUN_NAME>
```

where: 
- `<COUNTRY_CODE>` and `<ITER>` are defined above 
  
- `<MODE>`: there are two 10M random samples. One, which we'll call the "evaluation random sample" is only intended to evaluate the model's performance. The other one, which we'll call the "active learning random sample" is used to pick up and label new tweets through active learning.
  - `<MODE>` is equal to 0 if the inference is to be run on the evaluation random sample
    
  - `<MODE>` is equal to 1 if the inference is to be run on the active learning random sample
    
- `<RUN_NAME`: a name to differentiate output folders. The output folder name is defined as:
    - `iter_${ITER}-${RUN_NAME}-${SLURM_JOB_ID}-evaluation` if `<MODE> = 0`
    - `iter_${ITER}-${RUN_NAME}-${SLURM_JOB_ID}-new_samples` if `<MODE> = 1`




Note: 
-  This file runs the following command: 
```
$ python -u inference_ONNX_bert_100M_random.py --input_path ${INPUT_PATH} --output_path ${OUTPUT_MODELS} --country_code ${COUNTRY_CODE} --iteration_number ${ITER}
```

where:

- if `<MODE> = 0`: 
  - `INPUT_PATH=/scratch/mt4493/twitter_labor/twitter-labor-data/data/random_samples/random_samples_splitted/${COUNTRY_CODE}/evaluation` 
  - ` OUTPUT_PATH=/scratch/mt4493/twitter_labor/twitter-labor-data/data/inference/${COUNTRY_CODE}/iter_${ITER}-${RUN_NAME}-${JOB_ID}-evaluation
`
- if `<MODE> = 1`:
    - `  INPUT_PATH=/scratch/mt4493/twitter_labor/twitter-labor-data/data/random_samples/random_samples_splitted/${COUNTRY_CODE}/new_samples`
    - `OUTPUT_PATH=/scratch/mt4493/twitter_labor/twitter-labor-data/data/inference/${COUNTRY_CODE}/iter_${ITER}-${RUN_NAME}-${JOB_ID}-new_samples`
    

## Results:

### Inference files

For each label `<LABEL>`, all tweets with their respective score are saved at `<OUTPUT_PATH>/output/<LABEL>`. 

The logs are saved at `<OUTPUT_PATH>/logs/<LABEL>`.