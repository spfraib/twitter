#!/bin/bash

#SBATCH --job-name=cnnfiltered
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:0
#SBATCH --mail-type=END
#SBATCH --mail-user=nuclearr.d@gmail.com
#SBATCH --output=slurm_%j.out


module purge
singularity exec --nv /beegfs/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04.sif bash -c "                                                                                                              
source /scratch/spf248/pyenv_dval_wb_twitter_as_sam/py3.7/bin/activate                                                                                                                                             
cd /scratch/da2734/twitter/code/2-twitter_labor/4-inference_200M/
time python -u 10.4-GLOVE_CNN-deploying-100M_filtered_ONLY.py > /scratch/da2734/twitter/jobs/running_on_200Msamples/array_logs/GLOVE_CNN_FILTERED/${SLURM_ARRAY_TASK_ID}-$(date +%s).log 2>&1                          
exit                                                                                                                                                                                                               
"
