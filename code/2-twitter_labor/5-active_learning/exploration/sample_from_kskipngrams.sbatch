#!/bin/bash

#SBATCH --job-name=sample_kskipngrams
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=1
##SBATCH --gres=gpu:1
#SBATCH --output=slurm_sample_kskipngrams_%j.out
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu

COUNTRY_CODE=$1
MODEL_FOLDER=$2

# array=0-49

module purge
module load anaconda3/2020.02
source /scratch/mt4493/twitter_labor/code/envs/inference_env/bin/activate
cd /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/5-active_learning/exploration
LOG_FOLDER=log/${MODEL_FOLDER}
mkdir -p ${LOG_FOLDER}
srun time python -u sample_from_kskipngrams.py --country_code ${COUNTRY_CODE} --model_folder ${MODEL_FOLDER} > ${LOG_FOLDER}/sample_from_kskipngrams_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} 2>&1

exit

# sbatch --array=0-49 sample_from_kskipngrams.sbatch
