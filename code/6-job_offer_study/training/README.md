# Twitter-based study of demand-side on  labor market

## Annotation.py: 
File for generating tweets randonly for manual annotation

## convert_dataset_to_ner.py: 
File to convert job offer dataset to NER(CoNLL 2003) format.

## convert_conll_2columns.py: 
File to convert CoNll 2003 raw dataset to 2 columns (word, JOB_OFFER_TAG) and map I/B ORG and LOC tags to general ORG and LOC tags.

## finetune_ner.py: 
Finetune BERT base(cased) model using Conll 2003 dataset and job offer dataset using Simple Transformer and performing token level evaluation.
