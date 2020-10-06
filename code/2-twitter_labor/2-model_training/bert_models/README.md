# Fine-tuning BERT-based models

## Preliminary steps:

- Ideally, create a virtual environment specifically for training and activate it:

```
$ python3 -m virtualenv env_name
$ source env_name/bin/activate
```

- Install the necessary packages:
```
$ cd twitter/code/2-twitter_labor/model_training/bert_models #in case you haven't cd into the present folder yet
$ pip install -r requirements.txt
```

- Install PyTorch separately without cache to not use too much memory:
`$ pip install --no-cache-dir torch==1.5.0`

- Install [apex](https://github.com/nvidia/apex) to be able to use fp16 training. On Linux, it is done this way:
```
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

In case the first version causes mistakes, another possible solution to install apex is to replace the final line by:

`$ pip install -v --no-cache-dir ./`

We are now ready to start training.

## Training command:

To train a binary BERT-based classifier on all 5 classes on the cluster, run:

`$ sbatch train_bert_model.sbatch <DATA_FOLDER_NAME> <MODEL_NAME> <MODEL_TYPE> <INTRA_EPOCH_EVALUATION>`

with:
- <DATA_FOLDER_NAME>: the name of the folder in [twitter-labor-data/data](https://github.com/manueltonneau/twitter-labor-data/tree/master/data) where the train/val CSVs are stored (e.g. `jul23_iter0/preprocessed`)
- <MODEL_NAME>: the BERT-based model architecture used. By default, it is always set to `bert`. 
- <MODEL_TYPE>: the type of BERT-based model used (e.g. `DeepPavlov/bert-base-cased-conversational` for ConvBERT). This refers to the shortcut name of the model on the HuggingFace hub. The whole list can be found [here](https://huggingface.co/transformers/pretrained_models.html). 
- <INTRA_EPOCH_EVALUATION>: a string parsed as a boolean to determine whether to perform intra-epoch evaluation (10 per epoch by default). Possible values are `t` (parsed as `True`) or `f` (parsed as `False`)

The batch file can be found at: `/scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/2-model_training/bert_models`. 

Example command: 

`$ sbatch train_bert_model.sbatch jul23_iter0/preprocessed bert DeepPavlov/bert-base-cased-conversational t`
## Results:

### Models:

The trained models are automatically saved at: 

`/scratch/mt4493/twitter_labor/trained_models/${MODEL_TYPE}_${DATA_FOLDER_NAME}_${SLURM_JOB_ID}`.

A version of the model is saved for each epoch i at `${MODEL_TYPE}_${DATA_FOLDER_NAME}_${SLURM_JOB_ID}/${class}/models/checkpoint-${NB_STEPS}-epoch-${i}` where `class` is in `['lost_job_1mo', 'is_hired_1mo', 'job_search', 'job_offer', 'is_unemployed']`. 

The best model in terms of evaluation loss is saved in the folder: `${MODEL_TYPE}_${DATA_FOLDER_NAME}_${SLURM_JOB_ID}/${class}/models/best_model`. 

### Performance metrics and scores:

For each class and training round, four CSVs are saved:
- one with performance metrics (Precision/Recall/F1/AUC) on evaluation set, including metadata such as job_id, date and time, path of data used. It is called `val_${class}_evaluation.csv`. 
- one with performances metrics (//) on holdout set ( // ). It is called `holdout_${class}_evaluation.csv`. 
- one with scores for the evaluation set. It is called `val_${class}_scores.csv`. 
- one with scores for the holdout set. It is called `holdout_${class}_scores.csv`. 

Note that Precision/Recall/F1 are computed for a threshold of 0.5.

These four CSVs are saved at: `twitter_labor_data/data/${DATA_FOLDER_NAME}/results/<MODEL_TYPE>_<SLURM_JOB_IB>`. 
