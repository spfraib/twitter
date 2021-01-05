#!/bin/bash

#SBATCH --job-name=runjupyter
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

singularity exec --nv /beegfs/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04.sif /bin/bash
module purge
module load anaconda3/2020.02
source /scratch/da2734/pyenv_dval_wb_twitter/py3.7/bin/activate

port=$(shuf -i 10000-65500 -n 1)

/usr/bin/ssh -N -f -R $port:localhost:$port log-0
/usr/bin/ssh -N -f -R $port:localhost:$port log-1

cat<<EOF

Jupyter server is running on: $(hostname)
Job starts at: $(date)

Step 1 :

If you are working in NYU campus, please open an iTerm window, run command

ssh -L $port:localhost:$port $USER@prince.hpc.nyu.edu

If you are working off campus, you should already have ssh tunneling setup through HPC bastion host, 
that you can directly login to prince with command

ssh $USER@prince

Please open an iTerm window, run command

ssh -L $port:localhost:$port $USER@prince

Step 2:

Keep the iTerm windows in the previouse step open. Now open browser, find the line with

The Jupyter Notebook is running at: $(hostname)

the URL is something: http://localhost:${port}/?token=XXXXXXXX (see your token below)

you should be able to connect to jupyter notebook running remotly on prince compute node with above url

EOF

unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi

echo "imported conda, starting jupyter"
echo "shell" $0

export GENSIM_DATA_DIR=/scratch/da2734/twitter/code/11-baseline/logit_glove/downloaded

echo $GENSIM_DATA_DIR

jupyter notebook --no-browser --port $port --notebook-dir=$(pwd)




