#!/bin/bash

#SBATCH --job-name=scatter_plot
#SBATCH --nodes=1
#SBATCH --mem=40GB
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_scatter_plot_%j.out
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu


module purge
module load anaconda3/2020.02

source /scratch/mt4493/twitter_labor/code/envs/finetuning_bert_env/bin/activate

echo "pyenv activated"

cd /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/7-misc/understanding_quantization

python3 scatter_plot_torch_quantized.py

exit