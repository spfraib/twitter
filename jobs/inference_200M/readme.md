
# code to run onnx on 100M samples


## 1. convert pytorch models to onnx
python onnx_model_conversion.py /scratch/mt4493/twitter_labor/trained_models/iter0/jul23_iter0/DeepPavlov_bert-base-cased-conversational_jul23_iter0_preprocessed_11232989/{}/models/best_model 
note: the {} is important to be used to specify label of model AND make sure you have permisions to write to Manu's folder

## 2. run inference
sbatch --array=0-3000 ONYX_BERT_labels_deploy_100M_random_iteration_2.sbatch
note: 
1. this file runs the following command: python -u 10.7-ONNX-BERT-deploying-100M_random_ONLY_iteration2.py --model_path /scratch/mt4493/twitter_labor/trained_models/iter0/jul23_iter0/DeepPavlov_bert-base-cased-conversational_jul23_iter0_preprocessed_11232989/{}/models/best_model --output_path /scratch/mt4493/twitter_labor/code/twitter/jobs/inference_200M/inference_output/iteration2/output > /scratch/mt4493/twitter_labor/code/twitter/jobs/inference_200M/inference_output/iteration2/logs/${SLURM_ARRAY_TASK_ID}-$(date +%s).log 2>&1
2. for Bert, you need at least 5cpus for speeds ups. For glove, only 1 cpu is needed. 

## 2. check on progress
watch -n1 squeue -u da2734











sbatch --array=0-600 ONYX_BERT_labels_deploy_100M_random.sbatch

- you might be able to explore larger arrays - 1000 give me an slurm admin error, but I also get 162 machines max at the same time so it doens't matter much
- fyi the smallest memory I was able to use to run code was 5gb


# git clone manu's branch of simple transformers
git clone git@github.com:manueltonneau/simpletransformers.git
cd simpletransformers
pip install -e .

# do the same for transformers
https://github.com/manueltonneau/transformers.git

# make sure pytorch 1.5 is installed (other versions are weird)
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# install onyx
pip install onnxruntime_tools onnxruntime


(# my pyenv is here but don't use it while i'm using it for a job array as it can corrupt my job (happened with when I tried to use yours before)
(# source /scratch/da2734/pyenv_dval_wb_twitter/py3.7/bin/activate)