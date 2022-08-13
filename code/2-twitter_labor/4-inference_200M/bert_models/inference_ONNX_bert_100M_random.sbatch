#!/bin/bash

#SBATCH --job-name=ONNX
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu
#SBATCH --output=outfiles/slurm_inference_berts_%j.out

COUNTRY_CODE=$1
ITER=$2
MODE=$3
RUN_NAME=$4
METHOD=$5

SAM=1

JOB_ID=${SLURM_ARRAY_JOB_ID}

if [ ${SAM} -eq 1 ]
then
  SCRATCH_PATH=/scratch/spf248/scratch_manu
elif [ ${SAM} -eq 0 ]
then
  SCRATCH_PATH=/scratch/mt4493
fi

if [ ${MODE} -eq 0 ]
then
  INPUT_PATH=${SCRATCH_PATH}/twitter_labor/twitter-labor-data/data/random_samples/random_samples_splitted/${COUNTRY_CODE}/evaluation
  OUTPUT_PATH=${SCRATCH_PATH}/twitter_labor/twitter-labor-data/data/inference/${COUNTRY_CODE}/iter_${ITER}-${RUN_NAME}-${JOB_ID}-evaluation
elif [ ${MODE} -eq 1 ]
then
  INPUT_PATH=${SCRATCH_PATH}/twitter_labor/twitter-labor-data/data/random_samples/random_samples_splitted/${COUNTRY_CODE}/new_samples
  OUTPUT_PATH=${SCRATCH_PATH}/twitter_labor/twitter-labor-data/data/inference/${COUNTRY_CODE}/iter_${ITER}-${RUN_NAME}-${JOB_ID}-new_samples
elif [ ${MODE} -eq 2 ]
then
  INPUT_PATH=${SCRATCH_PATH}/twitter_labor/twitter-labor-data/data/random_samples/random_samples_splitted/${COUNTRY_CODE}/test
  OUTPUT_PATH=${SCRATCH_PATH}/twitter_labor/twitter-labor-data/data/inference/${COUNTRY_CODE}/iter_${ITER}-${RUN_NAME}-${JOB_ID}-test
fi

OUTPUT_MODELS=${OUTPUT_PATH}/output
OUTPUT_LOGS=${OUTPUT_PATH}/logs

# Create output folders if they don't exist
mkdir -p ${OUTPUT_MODELS}
mkdir -p ${OUTPUT_LOGS}


module purge
#module load anaconda3/2020.02
#
#source /scratch/mt4493/twitter_labor/code/envs/env_to_tar/inference_2021_env/bin/activate
#
#echo "pyenv activated"

cd ${SCRATCH_PATH}/twitter_labor/code/twitter/code/2-twitter_labor/4-inference_200M/bert_models

#if [ -z "${SLURM_ARRAY_TASK_ID}" ]
#then
#      echo SLURM_ARRAY_TASK_ID=test
#fi
echo 'running inference..'
singularity exec --overlay ${SCRATCH_PATH}/twitter_labor/code/envs/singularity/inference/env.ext3:ro \
	    /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python3 inference_ONNX_bert_100M_random.py \
	     --input_path ${INPUT_PATH} \
	     --output_path ${OUTPUT_MODELS} \
	     --country_code ${COUNTRY_CODE} \
	     --iteration_number ${ITER} \
	     --method ${METHOD}" #> ${OUTPUT_LOGS}/${SLURM_ARRAY_TASK_ID}-$(date +%s).log 2>&1
echo 'done'

exit



