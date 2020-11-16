#!/bin/bash

declare -a StringArray=('SV_8bS2BjePnBf6XaZ'
 'SV_3EtNuuwzOsM6yO1' 
 'SV_6JrZIYf8DQSo8F7' 
 'SV_eCX1MoB31N4OwO9' 
 'SV_cx9EpYoXVN3p1sx' 
 'SV_eWkEUtG1ktVE1jD' 
 'SV_2tKBgJaqDKz4VkF' 
 'SV_5nIw8CwF1v0128B' 
 'SV_9TabupxnmxtlyL3' 
 'SV_b78uXE4TG3vGwC1' 
 'SV_7VxaoBOsRCaZo9L' 
 'SV_cTG8uiAFsVPTOq9' 
 'SV_38GbM3bSN2lsq8J' 
 'SV_7a0R6QnfIYBZO7j' 
 'SV_b3NNaILgrFStdzL' 
 'SV_2lb5YiRSq9f59jf' 
 'SV_eEFLCnc2FT50QD3' 
 'SV_00wHtonmOcGzrx3' 
 'SV_8czHeb2J0p55UiN' 
 'SV_eD2m4MWNYdIwH2J')

DATA_PATH=/scratch/mt4493/twitter_labor/twitter-labor-data/data/qualtrics/US/labeling

cd DATA_PATH
if [ -f new_labels.pkl ]; then
  rm new_labels.pkl
fi


#for val in ${StringArray[@]}; do
#  if [ -d ${val} ]; then
#    rm -rf ${val}
#  fi
#done
#
#for val in ${StringArray[@]}; do
#    sbatch get_results_from_qualtrics.sbatch US ${val} 0 0
#    sleep(60)
#done

