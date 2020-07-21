#!/bin/bash
#SBATCH --job-name=training_${MODEL_TYPE}_${DATA_FOLDER}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_%j.out
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=manuel.tonneau@mailbox.org

cd /scratch/mt4493/twitter_labor/code/apex

pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
