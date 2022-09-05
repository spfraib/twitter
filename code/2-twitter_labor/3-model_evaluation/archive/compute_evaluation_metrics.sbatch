#!/bin/bash

#SBATCH --job-name=evaluation_metrics
#SBATCH --nodes=1
#SBATCH --mem=120GB
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm_evaluation_metrics_%j.out
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu

METHOD=$1
THRESHOLD=$2
TOPK=$3

module purge
module load anaconda3/2020.02

source /scratch/mt4493/twitter_labor/code/envs/inference_env/bin/activate

echo "pyenv activated"

cd /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/3-model_evaluation

python3 compute_evaluation_metrics.py \
--country_code US \
--method ${METHOD} \
--threshold ${THRESHOLD} \
--topk ${TOPK}

exit