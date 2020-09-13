# Structure of scratch/mt4493

# Note that:

# Description of active learning code:
What `code/2-twitter_labor/5-active_learning/select_tweets_to_label.py` doesCreate :
- Load all data from the random set (with operations such as tokenized text, etc..)
- Compute skipgrams (this can be outsourced to `preliminary/run_operations_on_initial_chunk.py` when we have figured out which `k` and `n` we use)
- Drop tweets if they are already labelled (i.e.: if their tweet ID appear in the training or evaluation set used at this iteration)
- Run the exploit part (the `method` parameter lets you decide between three types of method, defined in the parameter definition in the py file)
- Run the keyword exploration:
    - (Line 301) Drop stopwords and punctuation, calculate keyword lift and keep all words with lift > 1. Select `nb_top_lift_kw` keywords from these words.
    - (Line 303) Define the MLM pipeline (for now `bert-base-cased` as default, output 5x the number of keywords we want to use as output to make sure we have enough after discarding punctuation)