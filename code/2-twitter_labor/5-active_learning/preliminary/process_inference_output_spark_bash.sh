#!/bin/bash

INFERENCE_FOLDER=$1

module load python/gnu/3.6.5
module load spark/2.4.0
export PYSPARK_PYTHON=/share/apps/python/3.6.5/bin/python
export PYSPARK_DRIVER_PYTHON=/share/apps/python/3.6.5/bin/python
export PYTHONIOENCODING=utf8

hdfs dfs -put /scratch/mt4493/twitter_labor/twitter-labor-data/data/inference/${INFERENCE_FOLDER}/output /user/mt4493/twitter/inference/${INFERENCE_FOLDER}
echo "Loaded inference data on Hadoop. Launching the PySpark script"
CODE_FOLDER=/scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/5-active_learning/preliminary
TIMESTAMP=$(date +%s)
JOB_NAME=process_inference_output_${TIMESTAMP}
spark-submit --master yarn --deploy-mode cluster --name ${JOB_NAME} \
  --conf spark.yarn.submit.waitAppCompletion=false --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
  --conf spark.speculation=false --conf spark.yarn.appMasterEnv.LANG=en_US.UTF-8 \
  --conf spark.executorEnv.LANG=en_US.UTF-8 --driver-cores 10 \
  --driver-memory 10G --conf spark.dynamicAllocation.enabled=true --conf spark.dynamicAllocation.maxExecutors=50 \
  --executor-cores 10 --executor-memory 10G \
  ${CODE_FOLDER}/process_inference_output_spark.py \
  --inference_output_folder /user/mt4493/twitter/inference/${INFERENCE_FOLDER} \
  --random_chunks_with_operations_folder /user/mt4493/twitter/random_chunks_with_operations

echo "Submitted Spark job"

applicationId=$(yarn application -list -appStates RUNNING | awk -v tmpJob=${JOB_NAME} '{ if( $2 == tmpJob) print $1 }')
echo "$applicationId"
MINUTE_COUNT=0
while [ ! -z $applicationId ]; do
  echo "Waiting for job $JOB_NAME (application $applicationId) to be done to transfer output files to scratch. "
  echo "Already waited $MINUTE_COUNT minutes. "
  sleep 60
  MINUTE_COUNT=$((MINUTE_COUNT + 1))
  applicationId=$(yarn application -list -appStates RUNNING | awk -v tmpJob=${JOB_NAME} '{ if( $2 == tmpJob) print $1 }')
done
echo "Job is done. Copying data."
hdfs dfs -get /user/mt4493/twitter/inference/${INFERENCE_FOLDER}/joined /scratch/mt4493/twitter_labor/twitter-labor-data/data/inference/${INFERENCE_FOLDER}/output/joined
echo "Copying data finished."
