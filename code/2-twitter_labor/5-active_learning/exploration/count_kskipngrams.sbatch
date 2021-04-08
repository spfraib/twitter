#!/bin/bash

#SBATCH --job-name=count_kskipngrams
#SBATCH --nodes=1
#SBATCH --mem=360GB
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
##SBATCH --gres=gpu:1
#SBATCH --output=slurm_count_kskipngrams_%j.out
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu

COUNTRY_CODE=$1
MODEL_FOLDER=$2

module purge
module load anaconda3/2020.02
source /scratch/mt4493/twitter_labor/code/envs/kskipngrams_env/bin/activate
cd /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/5-active_learning/exploration
srun time python -u count_kskipngrams.py --country_code ${COUNTRY_CODE} --model_folder ${MODEL_FOLDER} > log/count_kskipngrams_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} 2>&1
exit

# sbatch --array=0-4 count_kskipngrams.sbatch