#!/bin/bash

#SBATCH --job-name=sample
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=500GB
#SBATCH --time=10:00:00
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu
#SBATCH --output=slurm_sample_%j.out

COUNTRY_CODE=$1

module purge
module load anaconda3/2020.02



