#!/bin/bash

#SBATCH --job-name=zip
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu
#SBATCH --output=slurm_zip_%j.out

COUNTRY_CODE=$1

cd /scratch/mt4493/twitter_labor/trained_models
tar --use-compress-program="pigz -k " -cf ${COUNTRY_CODE}.tar.gz ${COUNTRY_CODE}


