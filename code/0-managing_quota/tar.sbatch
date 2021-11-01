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

PATH=$1

module purge
module load anaconda3/2020.02

tar -zcvf ${PATH}.tar.gz ${PATH}