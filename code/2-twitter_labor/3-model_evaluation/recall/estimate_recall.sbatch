#!/bin/bash

#SBATCH --job-name=evaluation_metrics
#SBATCH --nodes=1
#SBATCH --mem=120GB
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm_estimate_recall_%j.out
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu

TOPK=$1

module purge
module load anaconda3/2020.02

source /scratch/mt4493/twitter_labor/code/envs/finetuning_bert_env/bin/activate

echo "pyenv activated"

cd /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/3-model_evaluation/recall

python3 estimate_recall.py \
--topk ${TOPK}

exit