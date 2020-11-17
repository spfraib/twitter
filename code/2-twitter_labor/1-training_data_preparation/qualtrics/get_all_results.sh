#!/bin/bash

COUNTRY_CODE=$1
DISCARD_X=$2

DATA_PATH=/scratch/mt4493/twitter_labor/twitter-labor-data/data/qualtrics/${COUNTRY_CODE}/labeling
CODE_PATH=/scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/1-training_data_preparation/qualtrics

if [ $COUNTRY_CODE == "US" ]; then
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

  cd ${DATA_PATH}

  if [ -f new_labels.pkl ]; then
    rm new_labels.pkl
    echo 'Removed new_labels.pkl'
  fi



  for val in ${StringArray[@]}; do
    if [ -d ${val} ]; then
      rm -rf ${val}
      echo "Removed ${val}"
    fi
  done

  cd ${CODE_PATH}
  for val in ${StringArray[@]}; do
      sbatch get_results_from_qualtrics.sbatch US ${val} 0 0 ${DISCARD_X}
      echo "Launched sbatch for ${val}"
      sleep 90
  done

else
  rm -rf ${DATA_PATH}
  mkdir ${DATA_PATH}
  echo "Removed the folder ${DATA_PATH} and recreated it"

  if [ ${COUNTRY_CODE} == "BR" ]; then
    declare -a StringArray=('SV_0ApchlyHQgsdkjP' 'SV_26xfsWKUoLzzMoJ' 'SV_3kD4BzmFl7iXNJz' 'SV_4Yjyub95bfMUPY1'
    'SV_5vtXIY1dRuOPJ2Z' 'SV_7VxaoBOsRCaZo9L' 'SV_8dgQSVhOq7lNSnz' 'SV_9uGO8SBIlfkjtHf' 'SV_agZdw3fjp1Z57mZ'
    'SV_cwEz2PXA0MsiudD' 'SV_egR4REW5E7DN45D')

  elif [ ${COUNTRY_CODE} == "MX" ]; then
    declare -a StringArray=('SV_00wHtonmOcGzrx3' 'SV_06CBdJBEJiaxEMt' 'SV_0GRfRbFbqV4kpJH' 'SV_4UEiYZ5y5ZTwcjr'
     'SV_6yScI3UEy8dGlj7' 'SV_8k59hFW37wsEPid' 'SV_9AzVa650Xypj8m9' 'SV_9RoaDghzhwCM21f' 'SV_9ucVxOvOk5Evs6F'
     'SV_9zwLWQ9nh1dzFkx' 'SV_bD6A4wCjqYumU9T' 'SV_br77pKyGnNbdjUN' 'SV_ctDmiVzThhbCZAF' 'SV_ei0ruhPkgtbNVTT'
     'SV_ebMrpYdD3tLLNl3')
  fi
  cd ${CODE_PATH}
  for val in ${StringArray[@]}; do
      sbatch get_results_from_qualtrics.sbatch ${COUNTRY_CODE} ${val} 0 0 ${DISCARD_X}
      echo "Launched sbatch for ${val}"
      sleep 90
  done
fi