# is_hired -> externship
# lost_job_1mo -> (kicked, out) 
# Change country_code + model_name
# Quick job with a few GB / no parallelization
from time import time
import pandas as pd
from glob import glob
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from decimal import Decimal
from transformers import AutoTokenizer
import re
import argparse

def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str)
    parser.add_argument("--model_folder", type=str)
    args = parser.parse_args()
    return args

args = get_args_from_command_line()
path_to_data = '/scratch/mt4493/twitter_labor/twitter-labor-data/data/active_learning'
path_to_fig = '/scratch/mt4493/twitter_labor/twitter-labor-data/data/active_learning/k_skip_n_grams/fig'
country_code = args.country_code
print('Country:', country_code)
model_name = args.model_folder
print('Model:', model_name)
motifs = ['1grams', '2grams', '3grams']
classes = ['is_hired_1mo', 'is_unemployed', 'job_offer', 'job_search', 'lost_job_1mo']
min_obs = 10
min_lift = 1
max_motifs = 5
existing_ngrams = {
'US':{
'1grams':['unemployed', 'jobless', 'unemployment', 'job', 'hiring', 'opportunity', 'apply'],
'2grams':[('i', 'fired'), ('laid', 'off'), ('found', 'job'), ('just', 'hired'), ('got', 'job'), ('started', 'job'), ('new', 'job'), ('i', 'unemployed'), ('i', 'jobless'), ('anyone', 'hiring'), ('wish', 'job'), ('need', 'job'), ('searching', 'job'), ('looking', 'gig'), ('applying', 'position'), ('find', 'job')],
'3grams':[('i', 'got', 'fired'), ('just', 'got', 'fired'), ('lost', 'my', 'job'), ('i', 'got', 'hired')],},
'BR':{
'1grams':[],
'2grams':[('perdi', 'emprego'), ('perdi', 'trampo'), ('fui', 'demitido'), ('me', 'demitiram'), ('consegui', 'emprego'), ('fui', 'contratado'), ('fui', 'contratada'), ('começo', 'emprego'), ('novo', 'emprego'), ('emprego', 'novo'), ('novo', 'trampo'), ('estou', 'desempregado'), ('estou', 'desempregada'), ('gostaria', 'emprego'), ('queria', 'emprego'), ('preciso', 'emprego'), ('procurando', 'emprego'), ('enviar', 'curriculo'), ('enviar', 'currículo'), ('envie', 'curriculo'), ('envie', 'currículo'), ('oportunidade', 'emprego'), ('temos', 'vagas')],
'3grams':[('me', 'mandaram', 'embora'), ('eu', 'sem', 'emprego')],},
'MX':{
'1grams':['nini', 'empleo', 'contratando', 'vacante'],
'2grams':[('me', 'despidieron'), ('me', 'corrieron'), ('consegui', 'empleo'), ('nuevo', 'trabajo'), ('nueva', 'chamba'), ('encontre', 'trabajo'), ('empiezo', 'trabajar'), ('estoy', 'desempleado'), ('sin', 'empleo'), ('sin', 'chamba'), ('necesito', 'trabajo'), ('busco', 'trabajo'), ('buscando', 'trabajo'), ('alguien', 'trabajo'), ('necesito', 'empleo'), ('empleo', 'nuevo'), ('estamos', 'contratando')],
'3grams':[('perdi', 'mi', 'trabajo'), ('perdí', 'mi', 'trabajo'), ('no', 'tengo', 'trabajo'), ('no', 'tengo', 'empleo'), ('no', 'tengo', 'chamba')],},
}[country_code]

model_dict = {'US': 'DeepPavlov/bert-base-cased-conversational', 'MX': 'dccuchile/bert-base-spanish-wwm-cased', 'BR': 'neuralmind/bert-base-portuguese-cased' }
tokenizer = AutoTokenizer.from_pretrained(model_dict[country_code])


for motif in motifs:
    print('# sampled',motif, ':', len(existing_ngrams[motif]))    
model_iter = int(re.findall('(?<=iter_).*?(?=-)',model_name)[0])
print('Iteration', model_iter)
if model_iter:
    for filename in sorted(glob(os.path.join(path_to_data,'k_skip_n_grams',country_code,'*','kskipngrams_*.json'))):
        filename_iter = int(re.findall('(?<=iter_).*?(?=-)',filename)[0])
        if filename_iter<model_iter:
            print('Remove ngrams sampled up to iteration', filename_iter)
            previous_ngrams = pd.read_json(filename).applymap(lambda x:tuple(x))
            for motif in motifs:
                existing_ngrams[motif].extend(previous_ngrams.filter(regex=motif).stack().tolist())
                print('# sampled',motif, ':', len(existing_ngrams[motif]))


# In[4]:


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

def select_motifs(df,min_lift=min_lift,min_obs=min_obs,max_motifs=max_motifs):
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


# In[5]:


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


# In[6]:


selected_motifs = pd.DataFrame({key:select_motifs(df[key]) for key in df})
os.makedirs(os.path.join(path_to_data,'k_skip_n_grams',country_code,model_name), exist_ok=True)
selected_motifs.to_json(os.path.join(path_to_data,'k_skip_n_grams',country_code,model_name,'kskipngrams_'+model_name+'.json'))
#selected_motifs.T


# # Figures

# In[7]:


model_names = sorted([x.split('/')[-1] for x in glob(os.path.join(path_to_data,'k_skip_n_grams',country_code,'iter_*'))])
#pd.concat([pd.read_json(os.path.join(path_to_data,'active_learning',country_code,model_name,'kskipngrams_'+model_name+'.json')) for model_name in model_names],keys=model_names).to_csv(os.path.join(path_to_data,'active_learning',country_code,'kskipngrams.csv'))
#pd.concat([pd.read_json(os.path.join(path_to_data,'active_learning',country_code,model_name,'kskipngrams_'+model_name+'.json')) for model_name in model_names],1,keys=[x.split('-')[0] for x in model_names]).T


# In[8]:


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
    os.makedirs(os.path.join(path_to_fig, model_name),exist_ok=True)
    plt.savefig(os.path.join(path_to_fig, model_name, motif+'_'+class_+{True:'_selected',False:''}[is_selected]+'.jpeg'),dpi=300,bbox_inches='tight')


# In[9]:


print('Plot data...')
for class_ in classes:
    start_time = time()
    print()
    print(class_)
    for motif in motifs[1:]:
        print(motif)
        plot_k_skip_n_grams(class_,motif)
    print('Time taken:', round(time() - start_time,1), 'seconds') # 82

