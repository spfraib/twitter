#!/bin/bash

COUNTRY_CODE=$1
NB_CORES=20

module load python/gnu/3.6.5
module load spark/2.4.0
export PYSPARK_PYTHON=/share/apps/python/3.6.5/bin/python
export PYSPARK_DRIVER_PYTHON=/share/apps/python/3.6.5/bin/python
export PYTHONIOENCODING=utf8

CODE_FOLDER=/scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/1-training_data_preparation/preliminary/tweets_2020
TIMESTAMP=$(date +%s)
JOB_NAME=convert_orc_to_parquet_${TIMESTAMP}
spark-submit --master yarn --deploy-mode cluster --name ${JOB_NAME} \
  --conf spark.yarn.submit.waitAppCompletion=false --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
  --conf spark.speculation=false --conf spark.yarn.appMasterEnv.LANG=en_US.UTF-8 \
  --conf spark.executorEnv.LANG=en_US.UTF-8 --driver-cores ${NB_CORES} \
  --driver-memory ${NB_CORES}G --conf spark.dynamicAllocation.enabled=true --conf spark.dynamicAllocation.maxExecutors=50 \
  --executor-cores ${NB_CORES} --executor-memory ${NB_CORES}G \
  ${CODE_FOLDER}/convert_orc_to_parquet.py \
  --country_code ${COUNTRY_CODE}


echo "Submitted Spark job"
