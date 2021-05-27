#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import *


# In[ ]:


spark = SparkSession.builder.appName("").getOrCreate()
path_to_data = '/user/spf248/twitter/data'
country_code = 'US'
model_name = 'iter_0-convbert-1122153'
classes = ['is_hired_1mo', 'is_unemployed', 'job_offer', 'job_search', 'lost_job_1mo']
motifs = ['one_grams', 'two_grams', 'three_grams']


# In[ ]:


text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'url', 'date', 'number'],
    # terms that will be annotated
    annotate={"hashtag", "allcaps", "elongated", "repeated",
        'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens
    
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter", 
    
    # corpus from which the word statistics are going to be used 
    # for spell correction
    corrector="twitter", 
    
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)

pre_process_udf = F.udf(lambda x:text_processor.pre_process_doc(x), ArrayType(StringType()))

def match_tokens(x,tokens):
    return len(set(x).intersection(tokens))==len(tokens)

def match_tokens_udf(tokens):
    return F.udf(lambda x: match_tokens(x,tokens),BooleanType())


# In[4]:


key2tokens = {'two_grams_is_hired_1mo': [('got', 'prom'),
  ('my', 'permit'),
  ('car', 'finally'),
  ('chapter', 'start'),
  ('paid', 'tomorrow'),
  ('started', 'workout'),
  ('early', 'yay'),
  ('license', 'today'),
  ('first', 'survived'),
  ('finished', 'homework')],
 'three_grams_is_hired_1mo': [('finally', 'got', 'phone'),
  ('!', 'my', 'woot'),
  ('first', 'tomorrow', 'work'),
  ('day', 'ready', 'start'),
  ('off', 'starting', 'year'),
  ('keys', 'new', 'the'),
  ('home', 'just', 'long'),
  ('friday', 'i', 'paid'),
  ('a', 'today', 'yay'),
  ('at', 'back', 'gym')],
 'two_grams_is_unemployed': [('im', 'losing'),
  ('lost', 'voice'),
  ('am', 'heartless'),
  ('have', 'migraine'),
  ('depressed', 'now'),
  ('food', 'starving'),
  ('mood', 'shitty'),
  ('attack', 'having'),
  ('hungover', 'still'),
  ('need', 'stressed')],
 'three_grams_is_unemployed': [('i', 'lost', 'voice'),
  ('am', 'losing', 'my'),
  ('have', 'headache', 'now'),
  ('a', 'breakdown', 'having'),
  ('eat', 'starving', 'to'),
  ('attack', 'had', 'just'),
  ('!', 'hungry', 'im'),
  ('been', 'for', 'sick'),
  ('bored', 'need', 'something'),
  ('depressed', 'do', 'not')],
 'two_grams_job_offer': [('bd', 'text'),
  ('buyer', 'looking'),
  ('becoming', 'interested'),
  ('ba', 'dm'),
  ('used', 'wd'),
  ('<money>', 'offering'),
  ('intern', 'internship'),
  ('auto', 'sale'),
  ('chevrolet', 'for'),
  ('local', 'wedding')],
 'three_grams_job_offer': [('bd', 'call', 'in'),
  ('ba', 'me', 'text'),
  ('estate', 'for', 'looking'),
  ('free', 'is', 'offering'),
  ('at', 'sale', 'wd'),
  ('buyer', 'on', 'real'),
  ('computer', 'deal', 'this'),
  ('auto', 'sales', 'used'),
  ('added', 'check', 'we'),
  ('!', '<email>', 'interested')],
 'two_grams_job_search': [('any', 'recommendations'),
  ('am', 'suggestions'),
  ('anyone', 'borrow'),
  ('hobby', 'need'),
  ('anybody', 'tonight'),
  ('i', 'takers'),
  ('ideas', 'something'),
  ('asap', 'some'),
  ('hmu', 'someone'),
  ('soon', 'trip')],
 'three_grams_job_search': [('any', 'i', 'recommendations'),
  ('?', 'am', 'suggestions'),
  ('find', 'need', 'something'),
  ('anyone', 'borrow', 'have'),
  ('buy', 'know', 'where'),
  ('soon', 'to', 'trip'),
  ('a', 'hobby', 'new'),
  ('give', 'me', 'ride'),
  ('back', 'get', 'gym'),
  ('good', 'in', 'places')],
 'two_grams_lost_job_1mo': [('just', 'kicked'),
  ('lost', 'power'),
  ('banned', 'got'),
  ('pulled', 'today'),
  ('bed', 'fell'),
  ('phone', 'yesterday'),
  ('flipped', 'off'),
  ('house', 'locked'),
  ('dumped', 'my'),
  ('blacked', 'i')],
 'three_grams_lost_job_1mo': [('got', 'just', 'kicked'),
  ('called', 'off', 'work'),
  ('lost', 'my', 'voice'),
  ('i', 'pulled', 'today'),
  ('house', 'locked', 'of'),
  ('for', 'hospital', 'out'),
  ('last', 'me', 'phone'),
  ('all', 'mad', 'over'),
  ('home', 'sent', 'to'),
  ('a', 'make', 'yesterday')]}


# In[ ]:


tweets = spark.read.parquet(os.path.join(path_to_data,'random_samples','random_samples_splitted',country_code,'new_samples'))
tweets = tweets.withColumn('tokens',pre_process_udf('text'))


# In[ ]:


for motif in motifs[1:]:
    for class_ in classes:
        key = motif+'_'+class_
        for tokens in key2tokens[key]:
            tmp = tweets.withColumn('matched_tokens',match_tokens_udf(tokens)('tokens'))
            tmp = tmp.filter(tmp['matched_tokens']==True).drop('tokens','matched_tokens').sample(False, 1.0).limit(100)
            tmp = tmp.withColumn('tokens', F.array([F.lit(x) for x in tokens]))
            tmp.write.mode("append").parquet(os.path.join(path_to_data,'active_learning',country_code,model_name,class_,motif))

