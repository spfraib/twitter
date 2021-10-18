#!/bin/bash

#SBATCH --gres=gpu:0
#SBATCH --job-name=demo_infer
#SBATCH --nodes=1
#SBATCH --mem=20GB
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --output=outfiles/slurm_demo_%j.out
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu

COUNTRY_CODE=US

module purge
singularity exec --bind /home/spf248/resolv.conf:/etc/resolv.conf:ro \
  --overlay /scratch/spf248/twitter/code/singularity/m3_inference.ext3:ro \
  /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
  /bin/bash -c "source /ext3/env.sh; python3 /scratch/spf248/twitter/code/twitter/code/5-demographics/inference/demographic_inference.py --country_code ${COUNTRY_CODE}"