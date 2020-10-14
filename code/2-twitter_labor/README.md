## Twitter-based labor market analysis

This folder contains code for a Twitter-based labor market analysis. It is organized as follows:

- `1-keyword_based_analysis`: count mentions of specific keywords within each tweet and use this count to predict the labor market situation.
- `2-model_training`: train both GloVe-CNN and BERT-based models for tweet binary classification on labor-market-related labels
- `3-model_evaluation`: evaluation of the trained models
- `4-inference_200M`: use trained models to do inference on a 200M sample of tweets
- `5-active_learning`: select next tweets to be labelled following an active learning approach (WIP)
- `6-bert_pretraining`: further pretrain BERT on a tweet corpus (WIP)

**Note:** the former folder entitled `7-classification` can be found in the `2-model_training/archive` folder. It contains code that does the following:
- Compute similarity of each tweet to a given sentence, allowing to find semantically similar tweets

- Feed a sample of tweets into a Qualtrics survey to create labels on Amazon Mechanical Turk

- Classify tweets based on the labor market status of Twitter users (outdated)
