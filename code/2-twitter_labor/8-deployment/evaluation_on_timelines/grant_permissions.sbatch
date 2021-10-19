#!/bin/bash

#SBATCH --job-name=grant_permissions
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3GB
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu
#SBATCH --output=jobarray_permissions_%j.out

cd /scratch/spf248/twitter/data
setfacl -R -m u:mt4493:rwx user_timeline/
setfacl -R -m u:mt4493:rwx labor