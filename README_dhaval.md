# Overview of where we're at:

- For active learning:
    - The commands to run only once (cf [README](https://github.com/spfraib/twitter/tree/manu_active_learning/code/2-twitter_labor/5-active_learning#commands-to-run-only-once)) are both done.
    - The command to run before each iteration (cf [README](https://github.com/spfraib/twitter/tree/manu_active_learning/code/2-twitter_labor/5-active_learning#to-run-after-each-iteration) ) has been launched on my account but hasn't started yet. You need this to do active learning.
    - I have detailed the active learning command in the [README](https://github.com/spfraib/twitter/tree/manu_active_learning/code/2-twitter_labor/5-active_learning#active-learning-1). To launch this, you will first have to define all the parameters in the [batch file](https://github.com/spfraib/twitter/blob/manu_active_learning/code/2-twitter_labor/5-active_learning/select_tweets_to_label.sbatch).
    - I have not merged the `manu_active_learning` branch in case there are other modifications to make. The up-to-date code is on `manu_active_learning`.
    - I haven't been able to test my [active learning code](https://github.com/spfraib/twitter/blob/manu_active_learning/code/2-twitter_labor/5-active_learning/select_tweets_to_label.py). I have written a detailed explanation of the code line per line though, that you can find down below.
    - Different active learning methods could potentially pick up the same tweets. We will have to drop duplicates at the end of the py file to make sure we don't send two times the same tweet to labelling (not yet implemented). 

- I haven't been able to implement the `further-fine-tuning`. I have found how to make it work though. You just have to modify the name of the folder you import the fine-tuned model from, to then further fine-tune it (see [this issue](https://github.com/ThilinaRajapakse/simpletransformers/issues/428)).
- There is not yet a complete agreement on the whole methodology. I list below a few points of disagreement:
    - The exploit part: Sam presented three possibilities for this, I implemented them all, see code documentation for more info.
    - You mentioned you only wanted to pick models as best_model at every epoch finish. We can discuss this but this is not how best_models are picked up so far (so far, best_model = model with lowest evaluation loss at or after the first epoch.)
    - For now, MLM is done with `bert-base-cased` since the results were really poor both with only pre-trained and fine-tuned ConvBERT. I opened [an issue](https://github.com/huggingface/transformers/issues/7007) in `huggingface/transformers` to figure out where that could come from. 
    - How to number iterations: I present what I think is best in the second bullet point of [this message](https://bigdatafusion.slack.com/archives/G012SA2NH51/p1599829669005600)
    - I haven't tested whether the new way of building training/evaluation sets (see point 3 of Active Learning methodology updated on page 2 of this [GDocs](https://docs.google.com/document/d/13tQQOjCHbUg_jJ_fGHeplJX5TwliqcQ_dQwrv7KoHPc/edit)) impacts performance. It might be interesting to check. 
    - In keyword exploration, when calculating lift, Sam wanted to discard that appear less than 10K times.
    - Sam not sure about usefulness of sentence exploration 
- From where we are, the next steps are:
    - Agree on methodology (including parameter values for active learning, all of these parameters are detailed in the py file)
    - Make sure the active learning script does exactly what the methodology suggests. Potential debugging.
    - Send tweets to labelling 
    - (Start of iteration 1 as defined above) Use new tweets for further fine-tuning. 
        - I talked to Sam about it and we would basically take the new tweets, do a train-test split and further fine-tuned on the training set. 
        - For training, everything is in `/code/2-twitter_labor/2-model_training`. We'll have to include further-finetuning in the code, as described above.
    - Do inference with this new further fine-tuned model. Code and README can be found in `/code/2-twitter_labor/4-inference_200M`
    - Select tweets to label through active learning. Code and README is in `/code/2-twitter_labor/5-active_learning`.
    - Have tweets labelled
    - And so on.
    



# Structure of scratch/mt4493

- Main folder is `/scratch/mt4493/twitter_labor`. 
- Github folder is in `twitter_labor/code/twitter`
- VEs are in `twitter_labor/code/envs`
- When training a model, models are saved in `twitter_labor/trained_models`
- The data is is stored in `/twitter_labor/twitter-labor-data`. It is linked with this [GitHub repo](https://github.com/manueltonneau/twitter-labor-data) even though the latter is not up-to-date.
    - both training/val but also evaluation results in `data/<DATE>_iter<ITERATION_NUMBER>`
    - inference output in `data/inference` folder
    - wordcounts in `data/wordcount_random`
     - chunks with operations in  `data/random_chunks_with_operations`
 

# Description of active learning code:
What `code/2-twitter_labor/5-active_learning/select_tweets_to_label.py` does for each class :
- Load all data from the random set (with operations such as tokenized text, etc..)
- Compute skipgrams (this can be outsourced to `preliminary/run_operations_on_initial_chunk.py` when we have figured out which `k` and `n` we use)
- Drop tweets if they are already labelled (i.e.: if their tweet ID appear in the training or evaluation set used at this iteration)
- Run the exploit part (the `method` parameter lets you decide between three types of method, defined in the parameter definition in the py file)
- Run the keyword exploration:
    - (Line 301) Drop stopwords and punctuation, calculate keyword lift and keep all words with lift > 1. Select top `nb_top_lift_kw` keywords from these words.
    - (Line 303) Define the MLM pipeline (for now `bert-base-cased` as default, output 5x the number of keywords we want to use as output to make sure we have enough after discarding punctuation)
    - (Line 310) Check whether the keywords picked by MLM appear in positives from the training set. Discard the ones that are in the training set.
    - (Line 313) Keep only top nb_kw_per_tweet_mlm keywords 
    - (Line 317) For each MLM keyword, take a random sample of size nb_tweets_per_kw_final of tweets that contain this MLM keyword. 
- Run the sentence exploration:
    - (Line 329) Determine the skipgram count on top tweets (in terms of base rate)
    - (Line 337) Discard all skipgrams that contain 2/3 or more of special tokens (results from preprocessing such as <hashtag> token) and/or punctuation. 
    - (Line 340-341) Take top nb_top_kskipngrams structures and store them in a list
    - (Line 350) For each top structure, sample nb_tweets_per_kskipngram tweets that contain this top structure. 
- All tweets to send to labelling are stored in the `tweets_to_label` pandas DataFrame. This dataframe has 6 columns:
    - `tweet_id`
    - `text`
    - `label`: for which class was this keyword picked up 
    - `source`: through which part of the active learning this tweet was picked up (`exploit`, `explore_keyword` or `explore_sentence` ).
    - `keyword`: in case `source = explore_keyword`, through which keyword 
    - `k-skip-n-gram`: in case `source = explore_sentence`, through which structure