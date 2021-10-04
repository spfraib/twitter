#!/bin/bash

#SBATCH --job-name=resize_tars
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5GB
#SBATCH --time=12:00:00
#SBATCH --output=slurm_resize_tars_%j.out
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu

COUNTRY_CODE=$1

module purge

TAR_INPUT_PATH=/scratch/spf248/twitter/data/demographics/profile_pictures/tars/${COUNTRY_CODE}
RESIZED_TAR_OUTPUT_PATH=/scratch/spf248/twitter/data/demographics/profile_pictures/resized_tars/${COUNTRY_CODE}

mkdir -p ${RESIZED_TAR_OUTPUT_PATH}

dir_count=$(ls ${TAR_INPUT_PATH}|wc -l)
loop_count=0
for tar_file in ${TAR_INPUT_PATH}/*;
do
    filename=${tar_file%.*}
    loop_count=$(($loop_count+1))
    if ! [[ ${filename} == *"err"* ]]; then
      tar -xf ${TAR_INPUT_PATH}/${tar_file} -C ${RESIZED_TAR_OUTPUT_PATH}/${filename}
      singularity exec --overlay /scratch/spf248/twitter/code/singularity/m3_inference.ext3:ro \
        /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
        /bin/bash -c "python3 /scratch/spf248/twitter/code/m3inference/scripts/preprocess.py --source_dir ${RESIZED_TAR_OUTPUT_PATH}/${filename} --output_dir ${RESIZED_TAR_OUTPUT_PATH}/resized_${filename}"
      mkdir empty_dir
      rsync -a --delete empty_dir/    ${RESIZED_TAR_OUTPUT_PATH}/${filename}/
      tar -cvf ${RESIZED_TAR_OUTPUT_PATH}/${filename}.tar ${RESIZED_TAR_OUTPUT_PATH}/resized_${filename}
      mkdir empty_dir
      rsync -a --delete empty_dir/    ${RESIZED_TAR_OUTPUT_PATH}/resized_${filename}/
    fi
    echo "Resized ${loop_count}/${dir_count} tar folders"
done