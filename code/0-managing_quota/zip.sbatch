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


PATH_TO_ZIP=$1

zip -r ${PATH_TO_ZIP}.zip ${PATH_TO_ZIP}

