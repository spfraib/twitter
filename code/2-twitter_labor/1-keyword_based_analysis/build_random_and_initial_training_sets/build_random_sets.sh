#!/bin/bash

COUNTRY_CODE=$1

module load python/gnu/3.6.5
module load spark/2.4.0
export PYSPARK_PYTHON=/share/apps/python/3.6.5/bin/python
export PYSPARK_DRIVER_PYTHON=/share/apps/python/3.6.5/bin/python
export PYTHONIOENCODING=utf8

hdfs dfs -mkdir -p /user/spf248/twitter/data/random_samples/${COUNTRY_CODE}/random_1
hdfs dfs -mkdir -p /user/spf248/twitter/data/random_samples/${COUNTRY_CODE}/random_2

CODE_FOLDER=/scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/1-keyword_based_analysis/build_random_and_initial_training_sets
TIMESTAMP=$(date +%s)
JOB_NAME=build_random_sets_${TIMESTAMP}
spark-submit --master yarn --deploy-mode cluster --name ${JOB_NAME} \
  --conf spark.yarn.submit.waitAppCompletion=false --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
  --conf spark.speculation=false --conf spark.yarn.appMasterEnv.LANG=en_US.UTF-8 \
  --conf spark.executorEnv.LANG=en_US.UTF-8 --driver-cores 30 \
  --driver-memory 30G --conf spark.dynamicAllocation.enabled=true --conf spark.dynamicAllocation.maxExecutors=50 \
  --executor-cores 30 --executor-memory 30G \
  ${CODE_FOLDER}/build_random_sets.py \
  --inference_output_folder /user/mt4493/twitter/inference/${INFERENCE_FOLDER}/output \
  --random_chunks_with_operations_folder /user/mt4493/twitter/random_chunks_with_operations


