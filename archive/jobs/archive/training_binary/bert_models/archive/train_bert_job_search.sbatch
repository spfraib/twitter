#!/bin/bash

#SBATCH --job-name=job_search
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

# when the job ends, send me an email at this email address.
# replace with your email address, and uncomment that line if you really need to receive an email.
#SBATCH --mail-type=END
#SBATCH --mail-user=nuclearr.d@gmail.com

# both standard output and standard error are directed to the same file.
# It will be placed in the directory I submitted the job from and will
# have a name like slurm_12345.out
#SBATCH --output=slurm_%j.out

module purge
module load anaconda3/2020.02
source /scratch/da2734/pyenv_dval_wb_twitter/py3.7/bin/activate
#module load jupyter-kernels/py2.7
#module load jupyter-kernels/py3.5
#module load miniconda


#/usr/bin/ssh -N -f -R $port:localhost:$port log-0
#/usr/bin/ssh -N -f -R $port:localhost:$port log-1


#unset XDG_RUNTIME_DIR
#if [ "$SLURM_JOBTMP" != "" ]; then
#    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
#fi


#source ~/miniconda3/bin/activate worldbank
#source activate worldbank
#conda activate worldbank

echo "pyenv activated"
#echo "shell" $0

cd /scratch/da2734/twitter/code/2-twitter_labor/2-model_training/
pwd

#jupyter notebook --no-browser --port $port --notebook-dir=$(pwd)
python -u training_binary_dhaval.py job_search > /scratch/da2734/twitter/jobs/training_binary/terminal_logs/train_bert_binary_separate_models_job_search.log 2>&1
#srun time python -u 8.2-random-samples-UNDERsampled-separate-labels.py job_search > /scratch/da2734/twitter/running_on_200Msamples/array_logs/job_search/8.2-random-samples-UNDERsampled-separate-labels_${SLURM_ARRAY_TASK_ID}-$(date +%s).log 2>&1
echo 'running'
exit



