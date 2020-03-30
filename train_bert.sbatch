#!/bin/bash

#SBATCH --job-name=training_binary
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=60GB
#SBATCH --time=168:00:00
#SBATCH --gres=gpu:1

# we want the job to be named "myMatlabTest" rather than something generated
# from the script name. This will affect the name of the job as reported
# by squeue:
#SBATCH --job-name=bert_trainer

# when the job ends, send me an email at this email address.
# replace with your email address, and uncomment that line if you really need to receive an email.
#SBATCH --mail-type=END
#SBATCH --mail-user=nuclearr.d@gmail.com

# both standard output and standard error are directed to the same file.
# It will be placed in the directory I submitted the job from and will
# have a name like slurm_12345.out
#SBATCH --output=slurm_%j.out

module purge
#module load jupyter-kernels/py2.7
module load jupyter-kernels/py3.5
module load miniconda

port=$(shuf -i 10000-65500 -n 1)

#/usr/bin/ssh -N -f -R $port:localhost:$port log-0
#/usr/bin/ssh -N -f -R $port:localhost:$port log-1


#unset XDG_RUNTIME_DIR
#if [ "$SLURM_JOBTMP" != "" ]; then
#    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
#fi

source ~/miniconda3/bin/activate worldbank
source activate worldbank
conda activate worldbank

echo "past source"
echo "shell" $0

cd /scratch/da2734/twitter/8-boundary-classified/

pwd

#jupyter notebook --no-browser --port $port --notebook-dir=$(pwd)
python -u training_binary.py > ../train_bert_binary_separate_models_.log 2>&1

exit



