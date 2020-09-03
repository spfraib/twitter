#!/bin/bash

#SBATCH --job-name=SAMoNNX
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=5GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:0
#SBATCH --mail-type=END
#SBATCH --mail-user=nuclearr.d@gmail.com
#SBATCH --output=slurm_%j.out

module purge
singularity exec --nv /beegfs/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04.sif \
    bash run-test.bash





