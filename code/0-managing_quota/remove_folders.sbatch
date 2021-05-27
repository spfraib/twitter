#!/bin/bash

#SBATCH --job-name=rm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu
#SBATCH --output=slurm_rm_%j.out

cd /scratch/mt4493/twitter_labor/twitter-labor-data/data/inference

rm -rf */*new_samples

