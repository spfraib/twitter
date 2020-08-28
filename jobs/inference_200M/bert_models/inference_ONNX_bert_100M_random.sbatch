#!/bin/bash

#SBATCH --job-name=ONNX
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=5GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu

MODEL_FOLDER=$1

MODEL_PATH=/scratch/mt4493/twitter_labor/trained_models/${MODEL_FOLDER}
JOB_ID=${SLURM_JOB_ID}
OUTPUT_PATH=/scratch/mt4493/twitter_labor/twitter-labor-data/data/inference/${MODEL_FOLDER}-${JOB_ID}
OUTPUT_MODELS=${OUTPUT_PATH}/output
OUTPUT_LOGS=${OUTPUT_PATH}/logs

# Create output folders if they don't exist
mkdir -p OUTPUT_MODELS
mkdir -p OUTPUT_LOGS

# both standard output and standard error are directed to the same file.
#SBATCH --output=slurm_inference_berts_%j.out

module purge
module load anaconda3/2019.10

source /scratch/mt4493/twitter_labor/code/envs/inference_env/bin/activate

echo "pyenv activated"
#echo "shell" $0

#export GENSIM_DATA_DIR=/scratch/da2734/twitter/code/11-baseline/logit_glove/downloaded

#echo $GENSIM_DATA_DIR

#echo what

cd /scratch/mt4493/twitter_labor/code/twitter/code/10-deploy-100M-samples/bert_models

pwd

echo 'running inference..'
srun time python -u inference_ONNX_bert_100M_random.py --model_path ${MODEL_PATH}/{}/models/best_model --output_path ${OUTPUT_MODELS} > ${OUTPUT_LOGS}/${SLURM_ARRAY_TASK_ID}-$(date +%s).log 2>&1
echo 'done'

exit



