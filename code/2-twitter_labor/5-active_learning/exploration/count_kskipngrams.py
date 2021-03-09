
# sbatch --array=0-4 count_kskipngrams.sh
import os
import sys
from time import time
from glob import glob
import numpy as np
import pandas as pd
import multiprocessing as mp
from itertools import combinations
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import argparse

# In[2]:


# Get Environment Variables
def get_env_var(varname,default):
    
    if os.environ.get(varname) != None:
        var = int(os.environ.get(varname))
        print(varname,':', var)
    else:
        var = default
        print(varname,':', var,'(Default)')
    return var

def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str)
    parser.add_argument("--model_folder", type=str)
    args = parser.parse_args()
    return args

SLURM_JOB_ID            = get_env_var('SLURM_JOB_ID',0)
SLURM_ARRAY_TASK_ID     = get_env_var('SLURM_ARRAY_TASK_ID',0)
SLURM_ARRAY_TASK_COUNT  = get_env_var('SLURM_ARRAY_TASK_COUNT',1)
SLURM_JOB_CPUS_PER_NODE = get_env_var('SLURM_JOB_CPUS_PER_NODE',mp.cpu_count())


# In[3]:

args = get_args_from_command_line()
path_to_input = '/scratch/mt4493/twitter_labor/twitter-labor-data/data'
path_to_output = '/scratch/mt4493/twitter_labor/twitter-labor-data/data/active_learning'
country_code = args.country_code
print('Country:', country_code)
model_name = args.model_folder
print('Model:', model_name)
class_ = ['is_hired_1mo', 'is_unemployed', 'job_offer', 'job_search', 'lost_job_1mo'][SLURM_ARRAY_TASK_ID]
print('Class:',class_)
motifs = ['1grams', '2grams', '3grams']
cutoff = 10000
print('Cutoff:', cutoff)
n_sample = 10**6
print('# random tweets:', n_sample)
rm = frozenset(['.','“','?','!',',',':','-','”','"',')','(','…','&','@','#','/','|',';','\`','\'','*','  ','’','t','\u200d','s','️','🏽','🏼','🏾','🏻','★','>','<','<percent>','<date>','<time>','</allcaps>', '<allcaps>', '<number>', '<repeated>', '<elongated>', '<hashtag>', '</hashtag>', '<url>', '<user>', '<email>'])


# In[4]:
corpus_dict = {'US': 'twitter',
               'MX': '/scratch/mt4493/twitter_labor/code/repos_annexe/ekphrasis/ekphrasis/stats/twitter_MX',
               'BR': '/scratch/mt4493/twitter_labor/code/repos_annexe/ekphrasis/ekphrasis/stats/twitter_BR'}

text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'url', 'date', 'number'],
#     # terms that will be annotated
#     annotate={"hashtag", "allcaps", "elongated", "repeated",
#         'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens
    
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter=corpus_dict[country_code],
    
    # corpus from which the word statistics are going to be used 
    # for spell correction
    corrector=corpus_dict[country_code],
    
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
#     dicts=[emoticons]
)


# In[5]:


def tokenize_text(text):
    tokens = text_processor.pre_process_doc(text)
    return [token for token in tokens if token not in rm]

def list_1grams(df_text):
    return list(map(tokenize_text, df_text)) 

def list_ngrams(df_1grams, n):
    return list(map(lambda x: list(combinations(x, n)), df_1grams)) 

def count_ngrams(df, motif):
    return df[motif].explode().reset_index().groupby(motif)['index'].count()


# In[6]:


print('Load tweets...')
start_time = time()
with mp.Pool() as pool:
    tweets = pd.concat(pool.map(pd.read_parquet, glob(os.path.join(path_to_input,'random_samples','random_samples_splitted',country_code,'new_samples','*.parquet'))))
tweets.reset_index(drop=True,inplace=True)
print('# tweets:', tweets.shape[0])
print('Time taken:', round(time() - start_time,1), 'seconds') # 82


# In[7]:


print('Select random tweets...')
start_time = time()
random_tweets = tweets.sample(n=n_sample, random_state=0).reset_index(drop=True)
print('# random tweets:', random_tweets.shape[0])
print('Time taken:', round(time() - start_time,1), 'seconds') # 4 (48 cores)


# In[8]:


print('Load scores...')
start_time = time()
with mp.Pool() as pool:
    scores = pd.concat(pool.map(pd.read_parquet, glob(os.path.join(path_to_input,'inference',country_code,model_name,'output',class_,'*.parquet'))))
print('# Scores:', scores.shape[0])
print('Time taken:', round(time() - start_time,1), 'seconds') # 49 (48 cores)


# In[9]:


print('Select top tweets...')
start_time = time()
top_tweets = scores.sort_values(by='score', ascending=False).reset_index().head(cutoff).merge(tweets, on='tweet_id')
print('# top tweets:',top_tweets.shape[0])
print('Time taken:', round(time() - start_time,1), 'seconds') # 162 (48 cores)


# In[10]:


print('Cleanup...')
start_time = time()
del tweets, scores
print('Time taken:', round(time() - start_time,1), 'seconds') # 162 (48 cores)


# In[11]:


start_time = time()
for df in [random_tweets, top_tweets]:
    print()
    print('Extract 1 grams')
    df['1grams'] = list_1grams(df['text'].values)
    for n in [2,3]:
        print('Extract',n,'grams')
        df[str(n)+'grams'] = list_ngrams(df['1grams'].values, n)
print('Time taken:', round(time() - start_time,1), 'seconds') # 645 (5 cores)


# In[12]:


ngrams = {}
print('Count ngrams...')
for motif in motifs:
    print()
    start_time = time()
    key = motif+'_random_'+str(n_sample)
    ngrams[key] = count_ngrams(random_tweets,motif).rename('n_random_'+str(n_sample))
    print('#',key,':', ngrams[key].shape[0])
    del random_tweets[motif]
    key = motif+'_top_'+class_
    ngrams[key] = count_ngrams(top_tweets,motif).rename('n_top_'+class_)
    ngrams[key] = ngrams[key].reset_index().merge(ngrams[motif+'_random_'+str(n_sample)].reset_index(),on=motif,how='left')
    ngrams[key]['lift_top_'+class_] = (ngrams[key]['n_top_'+class_]/cutoff)/(ngrams[key]['n_random_'+str(n_sample)]/n_sample)
    ngrams[key].sort_values(by='lift_top_'+class_,ascending=False,inplace=True)
    ngrams[key].set_index(motif,inplace=True)
    os.makedirs(os.path.join(path_to_output,'k_skip_n_grams',country_code,model_name,class_),exist_ok=True)
    ngrams[key].to_pickle(os.path.join(path_to_output,'k_skip_n_grams',country_code,model_name,class_,key+'.pkl'))
    print('#',key,':', ngrams[key].shape[0])
    del ngrams[key], ngrams[motif+'_random_'+str(n_sample)], top_tweets[motif]
    print('Time taken:', round(time() - start_time,1), 'seconds') # 


# In[12]:


print('Done !')

