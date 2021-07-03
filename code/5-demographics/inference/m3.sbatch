#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --job-name=demo_infer
#SBATCH --nodes=1
#SBATCH --mem=8GB
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_demo_%j.out

module purge
module load anaconda3/2020.02

source /scratch/spf248/joao/m3env/bin/activate

python demographic_inference.py 


exit

