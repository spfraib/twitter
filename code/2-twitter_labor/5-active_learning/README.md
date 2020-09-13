# Active Learning

## Preliminary:

### Commands to run only once:
Before starting the active learning methodology, we must:
- Run certain operations (tokenization, preprocessing, etc..) on the initial chunks of random set. **This must be done only once.** The following command will run these operations in parallel on all chunks and save the tweet and the operation results at `/scratch/mt4493/twitter_labor/twitter-labor-data/data/random_chunks_with_operations`:

`sbatch --array=0-824 preliminary/run_operations_on_initial_chunk.sbatch`

- Compute the word count of each word in the random set (will be needed for calculating keyword lift). The following command will use chunks in  `/scratch/mt4493/twitter_labor/twitter-labor-data/data/random_chunks_with_operations`, compute the word count for each word and save the result at `/scratch/mt4493/twitter_labor/twitter-labor-data/data/wordcount_random/wordcount_random.parquet`:

`sbatch create_wordcount_random.sbatch `

### To run after each iteration:
- Load all chunks outputted by inference (containing `tweet_id` and `score`) stored in the related `${INFERENCE_FOLDER}` in `/scratch/mt4493/twitter_labor/twitter-labor-data/data/inference`, sort by score and merge with chunks of random set (containing `tweet_id`, `text` and result of operations). For each `${CLASS}`, the following command will do this merge and save the parquet file `${CLASS}_all_sorted.parquet` combining all tweets, their score and the result of operations at `/scratch/mt4493/twitter_labor/twitter-labor-data/data/inference/${INFERENCE_FOLDER}/output/${CLASS}`:

`sbatch process_inference_output.sbatch ${INFERENCE_FOLDER}`

## Active learning:

After defining the hyperparameters in the `select_tweets_to_label.sbatch` file, run:

`sbatch select_tweets_to_label.sbatch ${INFERENCE_FOLDER} ${DATA_FOLDER}` 

where:

- `${INFERENCE_FOLDER}` refers to the name of the folder where the inference data is in `/scratch/mt4493/twitter_labor/twitter-labor-data/data/inference` (e.g. `DeepPavlov_bert-base-cased-conversational_jul23_iter0_preprocessed_12207397-12226078`)
- `${DATA_FOLDER}` refers to the name of the folder where the training/validation data is stored in `/scratch/mt4493/twitter_labor/twitter-labor-data/data` (e.g. `jul23_iter0/preprocessed`).

