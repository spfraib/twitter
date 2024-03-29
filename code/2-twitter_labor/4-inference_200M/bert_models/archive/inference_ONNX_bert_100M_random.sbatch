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
COUNTRY_CODE=$2
MODE=$3

MODEL_PATH=/scratch/mt4493/twitter_labor/trained_models/${MODEL_FOLDER}
JOB_ID=${SLURM_ARRAY_JOB_ID}


if [ ${MODE} -eq 0 ]
then
  INPUT_PATH=/scratch/mt4493/twitter_labor/twitter-labor-data/data/random_samples/random_samples_splitted/${COUNTRY_CODE}/evaluation
  OUTPUT_PATH=/scratch/mt4493/twitter_labor/twitter-labor-data/data/inference/${COUNTRY_CODE}/${MODEL_FOLDER}-${JOB_ID}-evaluation
elif [ ${MODE} -eq 1 ]
then
  INPUT_PATH=/scratch/mt4493/twitter_labor/twitter-labor-data/data/random_samples/random_samples_splitted/${COUNTRY_CODE}/new_samples
  OUTPUT_PATH=/scratch/mt4493/twitter_labor/twitter-labor-data/data/inference/${COUNTRY_CODE}/${MODEL_FOLDER}-${JOB_ID}-new_samples
fi

OUTPUT_MODELS=${OUTPUT_PATH}/output
OUTPUT_LOGS=${OUTPUT_PATH}/logs

# Create output folders if they don't exist
mkdir -p ${OUTPUT_MODELS}
mkdir -p ${OUTPUT_LOGS}

# both standard output and standard error are directed to the same file.
#SBATCH --output=slurm_inference_berts_%j.out

module purge
module load anaconda3/2020.02

source /scratch/mt4493/twitter_labor/code/envs/inference_env/bin/activate

echo "pyenv activated"
#echo "shell" $0

#export GENSIM_DATA_DIR=/scratch/da2734/twitter/code/11-baseline/logit_glove/downloaded

#echo $GENSIM_DATA_DIR

#echo what

cd /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/4-inference_200M/bert_models

pwd

echo 'running inference..'
srun time python -u inference_ONNX_bert_100M_random.py --model_path ${MODEL_PATH}/{}/models/best_model --input_path ${INPUT_PATH} --output_path ${OUTPUT_MODELS} --country_code ${COUNTRY_CODE} > ${OUTPUT_LOGS}/${SLURM_ARRAY_TASK_ID}-$(date +%s).log 2>&1
echo 'done'

exit



