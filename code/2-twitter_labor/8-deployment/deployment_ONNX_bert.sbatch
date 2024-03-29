#!/bin/bash

#SBATCH --job-name=deployment
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3GB
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu
#SBATCH --output=outfiles/jobarray_deployment_%j.out

COUNTRY_CODE=$1
ITER=$2
RESUME=$3
RESUME_FOLDER=$4

MODE=3
RUN_NAME=0
METHOD=0


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
elif [ ${MODE} -eq 3 ] && [ ${RESUME} -eq 0 ] # MEXICO inference
then
  INPUT_PATH=/scratch/spf248/twitter_labor_market_flows/data/text_labor_cls/${COUNTRY_CODE} #/iter${ITER}/unscored/
  OUTPUT_PATH=/scratch/spf248/twitter_labor_market_flows/data/bert_labor_cls/${COUNTRY_CODE}/iter${ITER}
elif [ ${MODE} -eq 3 ] && [ ${RESUME} -eq 1 ] # MEXICO inference
then
  INPUT_PATH=/scratch/spf248/twitter_labor_market_flows/data/text_labor_cls/${COUNTRY_CODE}
  OUTPUT_PATH=/scratch/spf248/twitter_labor_market_flows/data/bert_labor_cls/${COUNTRY_CODE}/${RESUME_FOLDER}
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
	    /bin/bash -c "source /ext3/env.sh; python /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/8-deployment/deployment_ONNX_bert.py --input_path ${INPUT_PATH} --output_path ${OUTPUT_MODELS} --country_code ${COUNTRY_CODE} --iteration_number ${ITER} --method ${METHOD} --resume ${RESUME} > ${OUTPUT_LOGS}/${SLURM_JOB_ID}.log"
#module load anaconda3/2020.02
#
#source /scratch/spf248/twitter/virtualenvs/mexico_virtualenv/bin/activate
#
#echo "pyenv activated"
#
#cd /scratch/spf248/twitter/code/twitter/code/2-twitter_labor/8-deployment
#
#pwd
#
#if [ -z "${SLURM_ARRAY_TASK_ID}" ]
#then
#      echo SLURM_ARRAY_TASK_ID=test
#fi
#echo 'running inference..'
##srun time python -u inference_ONNX_bert_MX.py --input_path ${INPUT_PATH} --output_path ${OUTPUT_MODELS} --country_code ${COUNTRY_CODE} --iteration_number ${ITER} --method ${METHOD} > ${OUTPUT_LOGS}/test.log 2>&1
#echo python -u deployment_ONNX_bert.py --input_path ${INPUT_PATH} --output_path ${OUTPUT_MODELS} --country_code ${COUNTRY_CODE} --iteration_number ${ITER} --method ${METHOD} --resume ${RESUME}
#srun time python -u deployment_ONNX_bert.py --input_path ${INPUT_PATH} --output_path ${OUTPUT_MODELS} --country_code ${COUNTRY_CODE} --iteration_number ${ITER} --method ${METHOD} --resume ${RESUME} > ${OUTPUT_LOGS}/${SLURM_JOB_ID}.log 2>&1
#
#echo 'done'
#
#exit



