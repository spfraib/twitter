#!/bin/bash

#SBATCH --job-name=merge_output
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30GB
#SBATCH --time=12:00:00
#SBATCH --output=slurm_merge_output_%j.out
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu

COUNTRY_CODE=$1

module purge
module load anaconda3/2020.02

cd /scratch/spf248/twitter/code/twitter/code/5-demographics/inference

python3 merge_output.py \
--country_code ${COUNTRY_CODE}