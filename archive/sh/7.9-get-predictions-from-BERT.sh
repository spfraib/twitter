#!/bin/bash
# This line tells the shell how to execute this script, and is unrelated 
# to SLURM.
   
# at the beginning of the script, lines beginning with "#SBATCH" are read by
# SLURM and used to set queueing options. You can comment out a SBATCH 
# directive with a second leading #, eg:
##SBATCH --nodes=1
   
# we need 1 node, will launch a maximum of one task. The task uses 2 CPU cores  
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
     
# we expect the job to finish within 1 hours. If it takes longer than 1
# hours, SLURM can kill it: 
#SBATCH --time=24:00:00
   
# we expect the job to use no more than 10GB of memory:
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
   
# we want the job to be named "myMatlabTest" rather than something generated 
# from the script name. This will affect the name of the job as reported
# by squeue: 
#SBATCH --job-name=predictions
 
# when the job ends, send me an email at this email address.
# replace with your email address, and uncomment that line if you really need to receive an email.
#SBATCH --mail-type=END
#SBATCH --mail-user=samuel.fraiberger@nyu.edu
   
# both standard output and standard error are directed to the same file.
# It will be placed in the directory I submitted the job from and will
# have a name like slurm_12345.out
#SBATCH --output=slurm_%j.out
 
# once the first non-comment, non-SBATCH-directive line is encountered, SLURM 
# stops looking for SBATCH directives. The remainder of the script is  executed
# as a normal Unix shell script
  
# first we ensure a clean running environment:
module purge
# and load the module for the software we are using:
#module load matlab/2016b
#module load mongodb/3.4.10
module load anaconda3/2020.02
source ~/mypython/py3.7/bin/activate

# the script will have started running in $HOME, so we need to move into the 
# directory we just created earlier
cd /scratch/spf248/twitter

# now start the job:
srun time python -u ./py/7.9-get-predictions-from-BERT.py > ./log/7.9-get-predictions-from-BERT-${SLURM_ARRAY_TASK_ID}-$(date +%s).log 2>&1
#cat thtest.m | srun matlab -nodisplay
# Leave a few empty lines in the end to avoid occasional EOF trouble.

exit
