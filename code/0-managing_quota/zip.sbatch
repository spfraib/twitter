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

cd /scratch/mt4493/twitter_labor/twitter-labor-data/data/inference

for i in */*new_samples ; do
  if [ -d "$i" ]; then
    zip -r $i.zip $i
  fi
done
