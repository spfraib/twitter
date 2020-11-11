#!/bin/bash

COUNTRY_CODE=$1
THIS_BATCH=$2

module purge
module load emoji/mt4493-20201106
module load spark/2.4.0
export PYSPARK_PYTHON=/share/apps/python/3.6.5/bin/python
export PYSPARK_DRIVER_PYTHON=/share/apps/python/3.6.5/bin/python
export PYTHONIOENCODING=utf8


CODE_FOLDER=/scratch/mt4493/twitter_labor/code/twitter/code/1-data_preparation/4-timelines
TIMESTAMP=$(date +%s)
JOB_NAME=update_users_timelines_${TIMESTAMP}
spark-submit --master yarn --deploy-mode cluster --name ${JOB_NAME} \
  --conf spark.yarn.submit.waitAppCompletion=false --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
  --conf spark.speculation=false --conf spark.yarn.appMasterEnv.LANG=en_US.UTF-8 \
  --conf spark.executorEnv.LANG=en_US.UTF-8 --driver-cores ${NB_CORES} \
  --driver-memory ${NB_CORES}G --conf spark.dynamicAllocation.enabled=true --conf spark.dynamicAllocation.maxExecutors=50 \
  --executor-cores ${NB_CORES} --executor-memory ${NB_CORES}G \
  ${CODE_FOLDER}/4.7-update-users-timelines-extract-pyspark.py \
  --country_code ${COUNTRY_CODE} \
  --this_batch ${THIS_BATCH}


echo "Submitted Spark job"
