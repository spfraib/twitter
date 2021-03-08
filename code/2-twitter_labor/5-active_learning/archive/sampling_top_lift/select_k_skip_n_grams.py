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


# In[2]:


path_to_data = '/scratch/spf248/twitter/data'
path_to_fig = '/scratch/spf248/twitter/fig'
country_code = 'US'
labels_name = 'jan5_iter0'
model_name = 'iter_0-convbert-1122153'
motifs = ['one_grams', 'two_grams', 'three_grams']
classes = ['is_hired_1mo', 'is_unemployed', 'job_offer', 'job_search', 'lost_job_1mo']
class2cutoff = {
'is_hired_1mo': 30338,
'is_unemployed': 21613,
'job_offer': 538490,
'job_search': 47970,
'lost_job_1mo': 2040}
min_obs = 10
min_lift = 50


# In[5]:


def load_and_preprocess(motif,class_):
    df = pd.read_pickle(os.path.join(path_to_data,'k_skip_n_grams',country_code,model_name,class_,motif+'_'+class_+'_top_'+str(class2cutoff[class_])+'.pkl'))
    df.rename(columns={[x for x in df.columns if 'n_labeled' in x][0]:'n_labeled',
                       [x for x in df.columns if '_random_' in x][0]:'n_random',
                       [x for x in df.columns if '_top_' in x and 'n_' in x][0]:'n_top',
                       [x for x in df.columns if '_top_' in x and 'lift_' in x][0]:'lift',
                       motif:'n_gram'},inplace=True)
    df['n_labeled'] = df['n_labeled'].replace(np.nan,0)
    df.set_index('n_gram',inplace=True)
    return df


# In[6]:


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
print('Time taken:', round(time() - start_time,1), 'seconds') # 82


# In[7]:


def plot_k_skip_n_grams(class_,motif,n_head=20):
    fig,ax = plt.subplots(figsize=(10,4))
    key = motif+'_'+class_
    data = df[key].loc[df[key]['n_random']>=min_obs].head(n_head)
    data['color'] = data['n_labeled'].apply(lambda x:'r' if not x else 'g')
    data['lift'].plot(ax=ax,kind='bar',color=data['color'])
    colors = {'Not present in labeled tweets':'r', 'Present in labeled tweets':'g'}         
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    plt.legend(handles, labels)
    ax.set_ylabel('Lift in top '+'%.0E' % Decimal(str(class2cutoff[class_]))+' tweets',fontweight='bold')
    ax.set_xlabel('')
    ax.tick_params(which='both',direction='in',pad=3)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45,ha='right')
    ax.locator_params(axis='y',nbins=5)
    ax.set_title(class_,fontweight='bold')  
    plt.savefig(os.path.join(path_to_fig,'k_skip_n_grams',motif+'_'+class_+'_selected.jpeg'),dpi=300,bbox_inches='tight')


# In[8]:


print('Plot data...')
for class_ in classes:
    start_time = time()
    print()
    print(class_)
    for motif in motifs[1:]:
        print(motif)
        plot_k_skip_n_grams(class_,motif)
    print('Time taken:', round(time() - start_time,1), 'seconds') # 82


# In[10]:


def select_motifs(df,min_lift=50,min_obs=10,remove_labeled=True):
    if min_lift:
        df = df[df['lift']>=min_lift]
    if min_obs:
        df = df[df['n_random']>=min_obs]
    if remove_labeled:
        df = df[df['n_labeled']==0]
    selected_motifs = []
    for new_motif in df.index:
        is_new = True
        for selected_motif in selected_motifs:
            if set(new_motif).intersection(selected_motif):
                is_new = False
                break
        if is_new:
            selected_motifs.append(new_motif)
    return df.loc[selected_motifs]


# In[12]:


for key in df:
    print(key)
    selected_df = select_motifs(df[key])
    selected_df.index = selected_df.index.map(list)
    selected_df.reset_index(inplace=True)
    os.makedirs(os.path.join(path_to_data,'k_skip_n_grams',country_code,model_name),exist_ok=True)
#     selected_df.to_parquet(os.path.join(path_to_data,'k_skip_n_grams',country_code,model_name,class_,key+'_filtered.parquet'))


# In[17]:


{key:select_motifs(df[key]).index.tolist()[:10] for key in df}


# In[14]:


def plot_k_skip_n_grams_selected(class_,motif,n_head=20):
    fig,ax = plt.subplots(figsize=(10,4))
    key = motif+'_'+class_
    data = select_motifs(df[key]).head(n_head)
    data['lift'].plot(ax=ax,kind='bar',color='k',alpha=.5)
    ax.set_ylabel('Lift in top '+'%.0E' % Decimal(str(class2cutoff[class_]))+' tweets',fontweight='bold')
    ax.set_xlabel('')
    ax.tick_params(which='both',direction='in',pad=3)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45,ha='right')
    ax.locator_params(axis='y',nbins=5)
    ax.set_title(class_,fontweight='bold')  
    plt.savefig(os.path.join(path_to_fig,'k_skip_n_grams',motif+'_'+class_+'_selected.jpeg'),dpi=300,bbox_inches='tight')


# In[15]:


print('Plot data...')
start_time = time()
for class_ in classes:
    
    print()
    print(class_)
    for motif in motifs[1:]:
        print(motif)
        plot_k_skip_n_grams_selected(class_,motif)
print('Time taken:', round(time() - start_time,1), 'seconds') # 82


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

