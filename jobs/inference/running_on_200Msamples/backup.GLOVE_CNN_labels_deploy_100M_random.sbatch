#!/bin/bash

#SBATCH --job-name=SAMoNNX
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=5GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:0

# when the job ends, send me an email at this email address.
# replace with your email address, and uncomment that line if you really need to receive an email.
#SBATCH --mail-type=END
#SBATCH --mail-user=nuclearr.d@gmail.com

# both standard output and standard error are directed to the same file.
# It will be placed in the directory I submitted the job from and will
# have a name like slurm_12345.out
#SBATCH --output=slurm_%j.out

#module purge
#module load anaconda3/2019.10
singularity exec --nv /beegfs/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04.sif /bin/bash
echo singularity
source /scratch/spf248/pyenv_dval_wb_twitter_as_sam/py3.7/bin/activate


#/usr/bin/ssh -N -f -R $port:localhost:$port log-0
#/usr/bin/ssh -N -f -R $port:localhost:$port log-1


#unset XDG_RUNTIME_DIR
#if [ "$SLURM_JOBTMP" != "" ]; then
#    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
#fi


echo "pyenv activated"
echo "shell" $0


cd /scratch/da2734/twitter/code/10-deploy-100M-samples/

pwd

srun time python -u 10.4-GLOVE_CNN-deploying-100M_random_ONLY.py > /scratch/da2734/twitter/jobs/running_on_200Msamples/array_logs/GLOVE_CNN_RANDOM/${SLURM_ARRAY_TASK_ID}-$(date +%s).log 2>&1

echo 'running'

exit



