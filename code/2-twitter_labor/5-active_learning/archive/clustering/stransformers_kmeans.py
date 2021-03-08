from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn import metrics
from argparse import Namespace
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow
import os
import matplotlib.pyplot as plt
from collections import Counter
from nltk import ngrams
import string
import re
import argparse

def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_folder", type=str,
                        default="iter_0-convbert-test-48-10-900-1538433-new_samples")
    parser.add_argument("--model_type", type=str,
                        help="Type of sentence transformers.",
                        default="paraphrase-xlm-r-multilingual-v1")
    parser.add_argument("--country_code", type=str,
                        default="US")
    parser.add_argument("--label", type=str)
    args = parser.parse_args()
    return args

def get_elbow_data(embeddings, k_max):
    distortions = []
    K = range(1,k_max)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(embeddings)
        distortions.append(kmeanModel.inertia_)
    return distortions

def get_elbow_graph(distortions, k_max):
    K = range(1,k_max)
    plt.figure(figsize=(16,8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

def perform_k_means(embeddings, num_clusters, corpus_df):
    clustering_model = KMeans(n_clusters=num_clusters)
    clustering_model.fit(embeddings)
    cluster_assignment = clustering_model.labels_
    clustered_sentences = [[] for i in range(num_clusters)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(corpus_df['text'].tolist()[sentence_id])
    return clustered_sentences, cluster_assignment

def clean_tweets(tweet_list):
    return [re.sub('http\S+', ' ', tweet_str) for tweet_str in tweet_list]

if __name__ == '__main__':
    # Define args from command line
    args = get_args_from_command_line()
    # Load model
    model = SentenceTransformer(args.model_type)
    # Define paths
    labor_data_path = '/scratch/mt4493/bert_twitter_labor/twitter-labor-data/data'
    data_path = f'{labor_data_path}/top_tweets/US/{args.inference_folder}'
    output_path = f'{labor_data_path}/evaluation_inference/clustering/US/{args.inference_folder}'
    embeddings_path = os.path.join(output_path, 'embeddings')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        os.makedirs(embeddings_path)
    # Load top tweets
    parquet_path = list(Path(os.path.join(data_path, args.label)).glob('*.parquet'))[0]
    top_tweets_df = pd.read_parquet(parquet_path)
    # Create/load embeddings
    embedding_file_path = os.path.join(embeddings_path, f'top_{args.label}_embeddings.npy')
    if not os.path.exists(embedding_file_path):
        text_list = clean_tweets(top_tweets_df['text'].tolist())
        corpus_embeddings = model.encode(text_list, convert_to_numpy=True)
        np.save(embedding_file_path, corpus_embeddings)
    else:
        corpus_embeddings = np.load(embedding_file_path)

