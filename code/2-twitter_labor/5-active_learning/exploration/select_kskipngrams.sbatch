#!/bin/bash

#SBATCH --job-name=select_kskipngrams
#SBATCH --nodes=1
#SBATCH --mem=10GB
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm_select_kskipngrams_%j.out
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu

COUNTRY_CODE=$1
MODEL_FOLDER=$2

module purge
module load anaconda3/2020.02
#source /scratch/mt4493/twitter_labor/code/envs/kskipngrams_env/bin/activate
source /scratch/mt4493/twitter_labor/code/envs/inference_env/bin/activate

cd /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/5-active_learning/exploration
python3 select_kskipngrams.py \
--country_code ${COUNTRY_CODE} \
--model_folder ${MODEL_FOLDER}

