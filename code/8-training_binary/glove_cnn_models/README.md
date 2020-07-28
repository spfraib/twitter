# Training GloVe + CNN models

## Preliminary steps:

- Ideally, create a virtual environment specifically for training and activate it:

```
$ python3 -m virtualenv env_name
$ source env_name/bin/activate
```

- Install the necessary packages:
```
$ cd twitter/code/8-training_binary/glove_cnn_models #in case you haven't cd into the present folder yet
$ pip install -r requirements.txt
```

We are now ready to start training.

## Training command:


To train a binary BERT-based classifier on all 5 classes on the cluster, run:

`$ sbatch train_glove_cnn_model.sbatch <DATA_FOLDER_NAME> `

with:
- <DATA_FOLDER_NAME>: the name of the folder in [twitter-labor-data/data](https://github.com/manueltonneau/twitter-labor-data/tree/master/data) where the train/val CSVs are stored (e.g. `jul23_iter0/preprocessed`)

The batch file can be found at: `/scratch/mt4493/twitter_labor/twitter/jobs/training_binary/glove_cnn_models/`. 

## Results:

### Models:

The trained models are automatically saved at: 

`/scratch/mt4493/twitter_labor/trained_models/GloVe_CNN_${DATA_FOLDER_NAME}_${SLURM_JOB_ID}`.

In the case of GloVe + CNN, only the best model in terms of evaluation loss is saved. It can be found in the folder: 

`GloVe_CNN_${DATA_FOLDER_NAME}_${SLURM_JOB_ID}/${class}/models/best_model`. 

### Performance metrics and scores:

For each class and training round, four CSVs are saved:
- one with performance metrics (Precision/Recall/F1/AUC) on evaluation set, including metadata such as job_id, date and time, path of data used. It is called `val_${class}_evaluation.csv`. 
- one with performances metrics (//) on holdout set ( // ). It is called `holdout_${class}_evaluation.csv`. 
- one with scores for the evaluation set. It is called `val_${class}_scores.csv`. 
- one with scores for the holdout set. It is called `holdout_${class}_scores.csv`. 

Note that Precision/Recall/F1 are computed for a threshold of 0.5.

These four CSVs are saved at: `twitter_labor_data/data/${DATA_FOLDER_NAME}/results/GloVe_CNN_<SLURM_JOB_IB>`. 
