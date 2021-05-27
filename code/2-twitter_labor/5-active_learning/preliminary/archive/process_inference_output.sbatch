#!/bin/bash

#SBATCH --job-name=merge_inference_output
#SBATCH --nodes=1
#SBATCH --mem=200GB
#SBATCH --output=slurm_merge_inference_output_%j.out
#SBATCH --time=96:00:00
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=nuclearr.d@gmail.com

INFERENCE_FOLDER=$1

module purge
module load anaconda3/2020.02

source /scratch/mt4493/twitter_labor/code/envs/inference_env/bin/activate
echo "pyenv activated"

cd /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/5-active_learning/preliminary

python3 -u process_inference_output.py --inference_output_folder /scratch/mt4493/twitter_labor/twitter-labor-data/data/inference/${INFERENCE_FOLDER}/output > /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/5-active_learning/process_inference_output.log 2>&1


