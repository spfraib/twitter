#!/bin/bash

#SBATCH --job-name=defer
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu
#SBATCH --output=slurm_defer_%j.out

NB_HRS_DELAY=$1
COMMAND=$2

sleep ${NB_HRS_DELAY}h

${COMMAND}