#!/bin/bash

#SBATCH --job-name=ONNX_conversion
#SBATCH --nodes=1
#SBATCH --mem=10GB
#SBATCH --gres=gpu:0
#SBATCH --output=slurm_ONNX_conversion_%j.out
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu

MODEL_FOLDER=$1
MODEL_TYPE=$2
MODEL_NAME=$3

module purge
module load anaconda3/2020.02

source /scratch/mt4493/twitter_labor/code/envs/inference_env/bin/activate

echo "pyenv activated"

cd /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/4-inference_200M/bert_models

python3 onnx_model_conversion.py \
--model_path /scratch/mt4493/twitter_labor/trained_models/${MODEL_FOLDER}/{}/models/best_model \
--model_type ${MODEL_TYPE} \
--model_name ${MODEL_NAME}

exit