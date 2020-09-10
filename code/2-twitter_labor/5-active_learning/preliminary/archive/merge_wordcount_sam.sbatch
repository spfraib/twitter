#!/bin/bash

#SBATCH --job-name=merge_wordcount_sam
#SBATCH --nodes=1
#SBATCH --mem=50GB
#SBATCH --gres=gpu:0
#SBATCH --output=slurm_wordcount_random_sam_%j.out
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu


module purge
module load anaconda3/2019.10

source /scratch/mt4493/twitter_labor/code/envs/inference_env/bin/activate
echo "pyenv activated"

cd /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/5-active_learning/preliminary

python3 merge_wordcount_sam.py