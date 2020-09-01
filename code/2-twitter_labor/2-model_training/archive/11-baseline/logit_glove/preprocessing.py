import pandas as pd
import re
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_file_path", type=str)
parser.add_argument("--output_folder", type=str)
args = parser.parse_args()

def cleaning(text_str):
    # Remove URL, RT, mention(@)
    text_str = text_str.replace(r'http(\S)+', r'')
    text_str = text_str.replace(r'http ...', r'')
    text_str = text_str.replace(r'(RT|rt)[ ]*@[ ]*[\S]+',r'')
    text_str = text_str.replace(r'@[\S]+',r'')
    # Remove non-ascii words or characters
    #text_str = [''.join([i if ord(i) < 128 else '' for i in text]) for text in text_str]
    # Remove extra space
    text_str = text_str.replace(r'[ ]{2, }', r' ')
    # Remove &, < and >
    text_str = text_str.replace(r'&amp;?', r'and')
    # Insert space between words and punctuation marks
    text_str = text_str.replace(r'([\w\d]+)([^\w\d ]+)', r'\1 \2')
    text_str = text_str.replace(r'([^\w\d ]+)([\w\d]+)', r'\1 \2')
    # Lowercased and strip
    text_str = text_str.lower()
    text_str = text_str.strip()
    return text_str

def text_length(text):
    return len(text.split(' '))

df = pd.read_csv(args.input_file_path, index_col=0, encoding='utf-8', engine='python')
#df.columns = ['text', 'InformationType_label']
df['ProcessedText'] = df['text'].apply(cleaning)
df['ProcessedText_length'] = df['ProcessedText'].apply(text_length)
print(os.path.join(args.output_folder,"preprocessed_" + os.path.split(args.input_file_path)[1]), flush=True)
df.to_csv(os.path.join(args.output_folder,"preprocessed_" + os.path.split(args.input_file_path)[1]))
