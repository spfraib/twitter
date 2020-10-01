# Active Learning

## Preliminary:

### Commands to run only once:
Before starting the active learning methodology, we must:
- Run certain operations (tokenization, preprocessing, etc..) on the initial chunks of random set. **This must be done only once.** The following command will run these operations in parallel on all chunks and save the tweet and the operation results at `/scratch/mt4493/twitter_labor/twitter-labor-data/data/random_chunks_with_operations`:

`sbatch --array=0-824 preliminary/run_operations_on_initial_chunk.sbatch`

- Compute the word count of each word in the random set (will be needed for calculating keyword lift). The following command will use chunks in  `/scratch/mt4493/twitter_labor/twitter-labor-data/data/random_chunks_with_operations`, compute the word count for each word and save the result at `/scratch/mt4493/twitter_labor/twitter-labor-data/data/wordcount_random/wordcount_random.parquet`:

`sbatch preliminary/create_wordcount_random.sbatch `

### To run after each iteration:

- For each class `${CLASS}`, load all chunks outputted by inference (containing `tweet_id`, `score`), join them all together and merge with the original tweets from the random containing the tweet `text` and the results of operations. Sort the whole dataset by inference score, save the top tweets (in terms of base rate) at `/scratch/mt4493/twitter_labor/twitter-labor-data/data/inference/${INFERENCE_FOLDER}/joined/${CLASS}/top_tweets_${CLASS}` and the joined chunks at `/scratch/mt4493/twitter_labor/twitter-labor-data/data/inference/${INFERENCE_FOLDER}/joined/${CLASS}/joined_chunks`. 

`sh preliminary/process_inference_output_spark_bash.sh ${INFERENCE_FOLDER}`

*Example command:*

`sh preliminary/process_inference_output_spark_bash.sh DeepPavlov_bert-base-cased-conversational_jul23_iter0_preprocessed_12207397-12226078`

**Note that this bash file must be run on the Dumbo cluster as it contains PySpark code.**

## Active learning (WIP):

After defining the hyperparameters in the `select_tweets_to_label.sbatch` file, run:

`sbatch select_tweets_to_label.sbatch ${INFERENCE_FOLDER} ${DATA_FOLDER}` 

where:

- `${INFERENCE_FOLDER}` refers to the name of the folder where the inference data is in `/scratch/mt4493/twitter_labor/twitter-labor-data/data/inference` (e.g. `DeepPavlov_bert-base-cased-conversational_jul23_iter0_preprocessed_12207397-12226078`)
- `${DATA_FOLDER}` refers to the name of the folder where the training/validation data is stored in `/scratch/mt4493/twitter_labor/twitter-labor-data/data` (e.g. `jul23_iter0/preprocessed`).



