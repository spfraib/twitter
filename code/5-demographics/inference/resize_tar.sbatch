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


TAR_INPUT_PATH=/scratch/spf248/twitter/data/demographics/profile_pictures/tars/${COUNTRY_CODE}
RESIZED_TAR_OUTPUT_PATH=/scratch/spf248/twitter/data/demographics/profile_pictures/resized_tars/${COUNTRY_CODE}

mkdir -p ${RESIZED_TAR_OUTPUT_PATH}

dir_count=$(ls ${TAR_INPUT_PATH}|wc -l)
loop_count=0
for tar_file in ${TAR_INPUT_PATH}/*;
do
    filename="$(basename -- $tar_file)"
    echo "Filename: ${filename}"
    filename_without_ext=${filename%.*}
    loop_count=$(($loop_count+1))
    if ! [[ ${filename} == *"err"* ]]; then
      mkdir -p ${RESIZED_TAR_OUTPUT_PATH}/${filename_without_ext}
      components=$(tar -tvf ${tar_file} | awk -F/ '$NF != "" { print $F; exit }' | tr -cd '/' | wc -c)
      components=$(($components-1))
      tar -xf ${tar_file} -C ${RESIZED_TAR_OUTPUT_PATH}/${filename_without_ext} --strip-components ${components}
      module purge
      singularity exec --overlay /scratch/spf248/twitter/code/singularity/m3_inference.ext3:ro \
        /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
        /bin/bash -c "source /ext3/env.sh; python3 /scratch/spf248/twitter/code/m3inference/scripts/preprocess.py --source_dir ${RESIZED_TAR_OUTPUT_PATH}/${filename_without_ext} --output_dir ${RESIZED_TAR_OUTPUT_PATH}/resized_${filename_without_ext}"
      mkdir -p empty_dir
      rsync -a --delete empty_dir/    ${RESIZED_TAR_OUTPUT_PATH}/${filename_without_ext}/
      rm -rf ${RESIZED_TAR_OUTPUT_PATH}/${filename_without_ext}/
      tar -cvf ${RESIZED_TAR_OUTPUT_PATH}/${filename_without_ext}.tar -C ${RESIZED_TAR_OUTPUT_PATH}/resized_${filename_without_ext} .
      mkdir -p empty_dir
      rsync -a --delete empty_dir/    ${RESIZED_TAR_OUTPUT_PATH}/resized_${filename_without_ext}/
      rm -rf ${RESIZED_TAR_OUTPUT_PATH}/resized_${filename_without_ext}/
    fi
    echo "Resized ${loop_count}/${dir_count} tar folders"
done