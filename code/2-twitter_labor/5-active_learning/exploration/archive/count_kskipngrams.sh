#!/bin/bash

#SBATCH --job-name=kskipngrams
#SBATCH --nodes=1
#SBATCH --mem=360GB
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
##SBATCH --gres=gpu:1
#SBATCH --output=slurm_count_kskipngrams_%j.out
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=samuel.fraiberger@nyu.edu


module purge
module load anaconda3/2020.02
source /scratch/mt4493/twitter_labor/code/envs/inference_env/bin/activate
cd /scratch/spf248/twitter
srun time python -u ./py/count_kskipngrams.py > ./log/count_kskipngrams_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} 2>&1
exit
