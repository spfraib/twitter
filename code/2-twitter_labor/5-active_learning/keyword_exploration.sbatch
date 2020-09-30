#!/bin/bash

#SBATCH --job-name=keyword_exploration
#SBATCH --nodes=1
#SBATCH --mem=20GB
#SBATCH --output=slurm_keyword_exploration_%j.out
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu

INFERENCE_FOLDER=$1
#DATA_FOLDER=$2

module purge
module load anaconda3/2019.10

source /scratch/mt4493/twitter_labor/code/envs/inference_env/bin/activate
echo "pyenv activated"

cd /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/5-active_learning

python3 -u keyword_exploration.py \
--inference_output_folder /scratch/mt4493/twitter_labor/twitter-labor-data/data/inference/${INFERENCE_FOLDER}/output \
--nb_top_lift_kw 10 \
> /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/5-active_learning/keyword_exploration.log 2>&1

