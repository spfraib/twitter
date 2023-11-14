#!/bin/bash

#SBATCH --job-name=deployment_US
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12GB
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu
#SBATCH --output=outfiles/jobarray_deployment_us_%j.out

CLASS=$1
RESUME=$2
RESUME_FOLDER=$3

COUNTRY_CODE=US
MODE=3
RUN_NAME=0
METHOD=0
ITER=0


JOB_ID=${SLURM_ARRAY_JOB_ID}
DATE=$(date '+%d%m%Y')

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
elif [ ${MODE} -eq 3 ] && [ ${RESUME} -eq 0 ] # MEXICO inference
then
#  INPUT_PATH=/scratch/spf248/twitter_labor_market_flows/data/bert_labor_cls/${COUNTRY_CODE}/iter${ITER}/${CLASS}/text_labor_cls
  INPUT_PATH=/scratch/spf248/twitter_labor_market_flows/data/user_timeline/US/31122022
  OUTPUT_PATH=/scratch/spf248/twitter_labor_market_flows/data/bert_labor_cls/${COUNTRY_CODE}/iter${ITER}/${CLASS}/${DATE}
#  INPUT_PATH=/scratch/spf248/twitter_labor_market_flows/data/text_labor_cls/${COUNTRY_CODE}/
#  OUTPUT_PATH=/scratch/spf248/twitter_labor_market_flows/data/tweet_scores/BERT/${COUNTRY_CODE}/iter_${ITER}-${SLURM_ARRAY_TASK_COUNT}-3GB-${JOB_ID}
elif [ ${MODE} -eq 3 ] && [ ${RESUME} -eq 1 ] # MEXICO inference
then
  INPUT_PATH=/scratch/spf248/twitter_labor_market_flows/data/user_timeline/US/31122022
  OUTPUT_PATH=/scratch/spf248/twitter_labor_market_flows/data/bert_labor_cls/${COUNTRY_CODE}/iter${ITER}/${CLASS}/${RESUME_FOLDER}
fi

OUTPUT_MODELS=${OUTPUT_PATH}/output
OUTPUT_LOGS=${OUTPUT_PATH}/logs #DEBUG
#OUTPUT_LOGS=/scratch/spf248/twitter/code/twitter/code/2-twitter_labor/8-deployment/

# Create output folders if they don't exist
mkdir -p ${OUTPUT_MODELS}
mkdir -p ${OUTPUT_LOGS}

echo ${INPUT_PATH}
echo ${OUTPUT_PATH}
echo ${OUTPUT_LOGS}
echo ${OUTPUT_MODELS}

module purge

singularity exec --overlay /scratch/mt4493/twitter_labor/code/envs/singularity/inference/env.ext3:ro \
	    /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/8-deployment/deployment_ONNX_bert.py --input_path ${INPUT_PATH} --output_path ${OUTPUT_MODELS} --country_code ${COUNTRY_CODE} --iteration_number ${ITER}_${CLASS} --method ${METHOD} --resume ${RESUME} > ${OUTPUT_LOGS}/${SLURM_JOB_ID}.log"
#module load anaconda3/2020.02



