#!/bin/bash

#SBATCH --job-name=wordcount_random
#SBATCH --nodes=1
#SBATCH --mem=250GB
#SBATCH --gres=gpu:0
#SBATCH --output=slurm_wordcount_random_%j.out
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu


module purge
module load anaconda3/2019.10

source /scratch/mt4493/twitter_labor/code/envs/inference_env/bin/activate
echo "pyenv activated"

cd /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/5-active_learning/preliminary

python3 create_wordcount_random.py

