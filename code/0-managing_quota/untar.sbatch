#!/bin/bash

#SBATCH --job-name=untar
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10GB
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu
#SBATCH --output=slurm_tar_%j.out

PATH_TO_UNTAR=$1

module purge

tar -zxvf ${PATH_TO_UNTAR}.tar.gz -C ${PATH_TO_UNTAR}