#!/bin/bash

#SBATCH --job-name=jupyter
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3GB
#SBATCH --gres=gpu:0
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=nuclearr.d@gmail.com
#SBATCH --output=slurm_inference_berts_%j.outs

module purge

port=$(shuf -i 10000-65500 -n 1)

/usr/bin/ssh -N -f -R $port:localhost:$port log-1
/usr/bin/ssh -N -f -R $port:localhost:$port log-2
/usr/bin/ssh -N -f -R $port:localhost:$port log-3

cat<<EOF

Jupyter server is running on: $(hostname)
Job starts at: $(date)

Step 1 :

If you are working in NYU campus, please open an iTerm window, run command

ssh -L $port:localhost:$port $USER@greene.hpc.nyu.edu

If you are working off campus, you should already have ssh tunneling setup through HPC bastion host, 
that you can directly login to greene with command

ssh $USER@greene

Please open an iTerm window, run command

ssh -L $port:localhost:$port $USER@greene

Step 2:

Keep the iTerm windows in the previouse step open. Now open browser, find the line with

The Jupyter Notebook is running at: $(hostname)

the URL is something: http://localhost:${port}/?token=XXXXXXXX (see your token below)

you should be able to connect to jupyter notebook running remotly on greene compute node with above url

EOF

unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi

if [[ $(hostname -s) =~ ^g ]]; then nv="--nv"; fi

jupyter notebook --no-browser --port $port --notebook-dir=$(pwd)





