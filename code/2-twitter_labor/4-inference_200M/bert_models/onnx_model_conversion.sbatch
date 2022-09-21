#!/bin/bash

#SBATCH --job-name=ONNX_conversion
#SBATCH --nodes=1
#SBATCH --mem=20GB
#SBATCH --gres=gpu:0
#SBATCH --time=10:00:00
#SBATCH --output=slurm_ONNX_conversion_%j.out
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu

COUNTRY_CODE=$1
ITER=$2
METHOD=$3

SAM=0

if [ ${SAM} -eq 1 ]
then
  SCRATCH_PATH=/scratch/spf248/scratch_manu
elif [ ${SAM} -eq 0 ]
then
  SCRATCH_PATH=/scratch/mt4493
fi


module purge
#module load anaconda3/2020.02
#
#source /scratch/mt4493/twitter_labor/code/envs/env_to_tar/inference_2021_env/bin/activate

#echo "pyenv activated"

singularity exec --overlay ${SCRATCH_PATH}/twitter_labor/code/envs/singularity/inference/env.ext3:ro \
	    /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python3 ${SCRATCH_PATH}/twitter_labor/code/twitter/code/2-twitter_labor/4-inference_200M/bert_models/onnx_model_conversion.py \
      --iteration_number ${ITER} \
      --country_code ${COUNTRY_CODE} \
      --method ${METHOD} \
      --scratch_path ${SCRATCH_PATH}"

exit