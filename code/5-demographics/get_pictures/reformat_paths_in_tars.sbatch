#!/bin/bash

#SBATCH --job-name=reformat_paths_in_tars
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5GB
#SBATCH --time=12:00:00
#SBATCH --output=slurm_reformat_paths_in_tars_%j.out
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu

COUNTRY_CODE=$1


TAR_INPUT_PATH=/scratch/spf248/twitter/data/demographics/profile_pictures/tars/${COUNTRY_CODE}


dir_count=$(ls ${TAR_INPUT_PATH}|wc -l)
loop_count=0
for tar_file in ${TAR_INPUT_PATH}/*;
do
    filename="$(basename -- $tar_file)"
    echo "Filename: ${filename}"
    filename_without_ext=${filename%.*}
    loop_count=$(($loop_count+1))
    if ! [[ ${filename} == *"err"* ]]; then
      components=$(tar -tvf ${tar_file} | awk -F/ '$NF != "" { print $F; exit }' | tr -cd '/' | wc -c)
      components=$(($components-1))
      echo "Components: ${components}"
      mkdir -p ${TAR_INPUT_PATH}/${filename_without_ext}
      tar -xf ${tar_file} -C ${TAR_INPUT_PATH}/${filename_without_ext} --strip-components ${components}
      mkdir -p empty_dir
      rsync -a --delete empty_dir/    ${tar_file}/
      rm -rf ${tar_file}
      tar -cvf ${tar_file} -C ${TAR_INPUT_PATH}/${filename_without_ext} .
      components=$(tar -tvf ${tar_file} | awk -F/ '$NF != "" { print $F; exit }' | tr -cd '/' | wc -c)
      components=$(($components-1))
      echo "Components after retarring: ${components}"
      rsync -a --delete empty_dir/    ${TAR_INPUT_PATH}/${filename_without_ext}/
      rm -rf ${TAR_INPUT_PATH}/${filename_without_ext}/
    fi
    echo "Retarred ${loop_count}/${dir_count} tar folders"
done

rm -rf empty_dir