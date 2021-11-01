#!/bin/bash

#SBATCH --job-name=tar
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu
#SBATCH --output=slurm_tar_%j.out

PATH_TO_TAR=$1

module purge

tar -zcvf ${PATH_TO_TAR}.tar.gz ${PATH_TO_TAR}