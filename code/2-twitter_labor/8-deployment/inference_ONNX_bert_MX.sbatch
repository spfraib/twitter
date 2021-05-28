#!/bin/bash

#SBATCH --job-name=mxinference
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu
#SBATCH --output=slurm_inference_berts_%j.out

COUNTRY_CODE=$1
ITER=$2
MODE=$3
RUN_NAME=$4
METHOD=$5

JOB_ID=${SLURM_ARRAY_JOB_ID}


if [ ${MODE} -eq 0 ]
then
  INPUT_PATH=/scratch/mt4493/twitter_labor/twitter-labor-data/data/random_samples/random_samples_splitted/${COUNTRY_CODE}/evaluation
  OUTPUT_PATH=/scratch/mt4493/twitter_labor/twitter-labor-data/data/inference/${COUNTRY_CODE}/iter_${ITER}-${RUN_NAME}-${JOB_ID}-evaluation
elif [ ${MODE} -eq 1 ]
then
  INPUT_PATH=/scratch/mt4493/twitter_labor/twitter-labor-data/data/random_samples/random_samples_splitted/${COUNTRY_CODE}/new_samples
  OUTPUT_PATH=/scratch/mt4493/twitter_labor/twitter-labor-data/data/inference/${COUNTRY_CODE}/iter_${ITER}-${RUN_NAME}-${JOB_ID}-new_samples
elif [ ${MODE} -eq 2 ]
then
  INPUT_PATH=/scratch/mt4493/twitter_labor/twitter-labor-data/data/random_samples/random_samples_splitted/${COUNTRY_CODE}/test
  OUTPUT_PATH=/scratch/mt4493/twitter_labor/twitter-labor-data/data/inference/${COUNTRY_CODE}/iter_${ITER}-${RUN_NAME}-${JOB_ID}-test
elif [ ${MODE} -eq 3 ] # MEXICO inference
then
  INPUT_PATH=/scratch/spf248/twitter/data/user_timeline/user_timeline_text_preprocessed/${COUNTRY_CODE}/
  OUTPUT_PATH=/scratch/spf248/twitter/data/user_timeline/bert_inferrred/${COUNTRY_CODE}/iter_${ITER}-${RUN_NAME}-${JOB_ID}
fi

OUTPUT_MODELS=${OUTPUT_PATH}/output
#OUTPUT_LOGS=${OUTPUT_PATH}/logs #DEBUG
OUTPUT_LOGS=/scratch/spf248/twitter/code/twitter/code/2-twitter_labor/8-deployment/

# Create output folders if they don't exist
mkdir -p ${OUTPUT_MODELS}
mkdir -p ${OUTPUT_LOGS}


module purge
module load anaconda3/2020.02

source /scratch/spf248/twitter/virtualenvs/mexico_virtualenv/bin/activate

echo "pyenv activated"

cd /scratch/spf248/twitter/code/twitter/code/2-twitter_labor/8-deployment

pwd

if [ -z "${SLURM_ARRAY_TASK_ID}" ]
then
      echo SLURM_ARRAY_TASK_ID=test
fi
echo 'running inference..'
#srun time python -u inference_ONNX_bert_MX.py --input_path ${INPUT_PATH} --output_path ${OUTPUT_MODELS} --country_code ${COUNTRY_CODE} --iteration_number ${ITER} --method ${METHOD} > ${OUTPUT_LOGS}/test.log 2>&1
srun time python -u inference_ONNX_bert_MX.py --input_path ${INPUT_PATH} --output_path ${OUTPUT_MODELS} --country_code ${COUNTRY_CODE} --iteration_number ${ITER} --method ${METHOD} > ${OUTPUT_LOGS}/${SLURM_ARRAY_TASK_ID}-$(date +%s).log 2>&1
#e.g.
#python -u inference_ONNX_bert_MX.py --input_path /scratch/spf248/twitter/data/user_timeline/user_timeline_text_preprocessed/MX/ --output_path /scratch/spf248/twitter/data/user_timeline/bert_inferrred/MX/ --country_code MX --iteration_number 6 --method 0
echo 'done'

exit



