#!/bin/bash

#SBATCH --job-name=ONNX_conversion
#SBATCH --nodes=1
#SBATCH --mem=10GB
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=manuel.tonneau@mailbox.org

MODEL_FOLDER=$1

module purge
module load anaconda3/2019.10

source /scratch/mt4493/twitter_labor/code/envs/inference_env/bin/activate

echo "pyenv activated"

cd /scratch/mt4493/twitter_labor/code/twitter/code/10-deploy-100M-samples/bert_models

python3 onnx_model_conversion.py /scratch/mt4493/twitter_labor/trained_models/${MODEL_FOLDER}/{}/models/best_model

exit