#!/usr/bin/env python
# coding: utf-8

# In[1]:


from time import time
import pandas as pd
from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
from transformers import AutoTokenizer


# In[2]:


path_to_data = '/scratch/spf248/twitter/data'
path_to_fig = '/scratch/spf248/twitter/fig'
country_code = 'US'
model_name = 'iter_0-convbert-1122153'
motifs = ['1grams', '2grams', '3grams']
classes = ['is_hired_1mo', 'is_unemployed', 'job_offer', 'job_search', 'lost_job_1mo']
min_obs = 10
min_lift = 1
existing_ngrams = {
    '1grams':['unemployed', 'jobless', 'unemployment', 'job', 'hiring', 'opportunity', 'apply'],
    '2grams':[('i', 'fired'), ('laid', 'off'), ('found', 'job'), ('just', 'hired'), ('got', 'job'), ('started', 'job'), ('new', 'job'), ('i', 'unemployed'), ('i', 'jobless'), ('anyone', 'hiring'), ('wish', 'job'), ('need', 'job'), ('searching', 'job'), ('looking', 'gig'), ('applying', 'position'), ('find', 'job')],
    '3grams':[('i', 'got', 'fired'), ('just', 'got', 'fired'), ('lost', 'my', 'job'), ('i', 'got', 'hired')],
}
tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/bert-base-cased-conversational')


# In[3]:


def is_in_vocab(tokens, tokenizer = tokenizer):
    tokenized_tokens = tokenizer.batch_decode(tokenizer.batch_encode_plus(list(tokens))['input_ids'])
    if not any(['[UNK]' in x for x in tokenized_tokens]):
        return True
    else:
        return False
    
def has_duplicates(tokens):
    return len(set(tokens))!=len(tokens)

def has_common_tokens(tokens,tokens_list):
    return sum([len(set(tokens).intersection(x)) for x in tokens_list])>0

def load_and_preprocess(motif,class_):
    df = pd.read_pickle(os.path.join(path_to_data,'k_skip_n_grams',country_code,model_name,class_,motif+'_top_'+class_+'.pkl'))
    df.rename({motif:'n_gram'}, inplace=True)
    df.rename(columns={[x for x in df.columns if 'n_random' in x][0]:'n_random',
                       [x for x in df.columns if 'n_top' in x][0]:'n_top',
                       [x for x in df.columns if 'lift' in x][0]:'lift'}, inplace=True)
    return df


# In[4]:


df = {}
start_time = time()
print('Load data...')
for class_ in classes:
    print()
    print(class_)
    for motif in motifs[1:]:
        print(motif)
        key = motif+'_'+class_
        df[key] = load_and_preprocess(motif,class_)
        print('#',motif,':',df[key].shape[0])
print('Time taken:', round(time() - start_time,1), 'seconds') # 82


# In[5]:


def select_motifs(df,min_lift=1,min_obs=10,max_motifs=10):
    if min_lift:
        df = df[df['lift']>=min_lift]
    if min_obs:
        df = df[df['n_random']>=min_obs]
    selected_motifs = []
    for new_motif in df.index:
        if is_in_vocab(new_motif)         and not has_common_tokens(new_motif, selected_motifs)         and not has_duplicates(new_motif)         and not new_motif in existing_ngrams[str(len(new_motif))+'grams']:
            selected_motifs.append(new_motif)
        if len(selected_motifs)>=max_motifs:
            break
    return selected_motifs


# In[12]:


selected_motifs = pd.DataFrame({key:select_motifs(df[key]) for key in df})
os.makedirs(os.path.join(path_to_data,'active_learning',country_code,model_name), exist_ok=True)
selected_motifs.to_pickle(os.path.join(path_to_data,'active_learning',country_code,model_name,'kskipngrams.pkl'), protocol = 4)
selected_motifs


# # Figures

# In[19]:


def plot_k_skip_n_grams(class_,motif,n_head=20,is_selected=True):
    fig,ax = plt.subplots(figsize=(8,4))
    key = motif+'_'+class_
    if is_selected:
        data = df[key].loc[select_motifs(df[key])].head(n_head)
    else:
        data = df[key].head(n_head)
    data['lift'].plot(ax=ax,kind='bar',color='k',alpha=.5)
    ax.set_ylabel('Lift in top tweets',fontweight='bold')
    ax.set_xlabel('')
    ax.tick_params(which='both',direction='in',pad=3)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45,ha='right')
    ax.locator_params(axis='y',nbins=5)
    ax.set_title(class_,fontweight='bold')  
    plt.savefig(os.path.join(path_to_fig, 'k_skip_n_grams',motif+'_'+class_+{True:'_selected',False:''}[is_selected]+'.jpeg'),dpi=300,bbox_inches='tight')


# In[20]:


print('Plot data...')
for class_ in classes:
    start_time = time()
    print()
    print(class_)
    for motif in motifs[1:]:
        print(motif)
        plot_k_skip_n_grams(class_,motif)
    print('Time taken:', round(time() - start_time,1), 'seconds') # 82


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[45]:


df[key][df[key][motif]=='externship']


# In[52]:


df[key][df[key][motif]==('kicked', 'out')]


# In[15]:


df['three_grams_lost_job_1mo_top_10000'][df['three_grams_lost_job_1mo_top_10000']['three_grams'].apply(
lambda x:'kicked' in x and 'out' in x and 'work' in x)].head(50)

