"""
*****ADAPTED FROM https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/clustering/fast_clustering.py*****
This is a more complex example on performing clustering on large scale dataset.
This examples find in a large set of sentences local communities, i.e., groups of sentences that are highly
similar. You can freely configure the threshold what is considered as similar. A high threshold will
only find extremely similar sentences, a lower threshold will find more sentence that are less similar.
A second parameter is 'min_community_size': Only communities with at least a certain number of sentences will be returned.
The method for finding the communities is extremely fast, for clustering 50k sentences it requires only 5 seconds (plus embedding comuptation).
In this example, we download a large set of questions from Quora and then find
similar questions in this set.
"""
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
import csv
import pickle
import time
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def community_detection(embeddings, threshold=0.75, min_community_size=10, init_max_size=1000):
    """
    Function for Fast Community Detection
    Finds in the embeddings all communities, i.e. embeddings that are close (closer than threshold).
    Returns only communities that are larger than min_community_size. The communities are returned
    in decreasing order. The first element in each list is the central point in the community.
    """

    # Compute cosine similarity scores
    cos_scores = util.pytorch_cos_sim(embeddings, embeddings)

    # Minimum size for a community
    top_k_values, _ = cos_scores.topk(k=min_community_size, largest=True)

    # Filter for rows >= min_threshold
    extracted_communities = []
    for i in range(len(top_k_values)):
        if top_k_values[i][-1] >= threshold:
            new_cluster = []

            # Only check top k most similar entries
            top_val_large, top_idx_large = cos_scores[i].topk(k=init_max_size, largest=True)
            top_idx_large = top_idx_large.tolist()
            top_val_large = top_val_large.tolist()

            if top_val_large[-1] < threshold:
                for idx, val in zip(top_idx_large, top_val_large):
                    if val < threshold:
                        break

                    new_cluster.append(idx)
            else:
                # Iterate over all entries (slow)
                for idx, val in enumerate(cos_scores[i].tolist()):
                    if val >= threshold:
                        new_cluster.append(idx)

            extracted_communities.append(new_cluster)

    # Largest cluster first
    extracted_communities = sorted(extracted_communities, key=lambda x: len(x), reverse=True)

    # Step 2) Remove overlapping communities
    unique_communities = []
    extracted_ids = set()

    for community in extracted_communities:
        add_cluster = True
        for idx in community:
            if idx in extracted_ids:
                add_cluster = False
                break

        if add_cluster:
            unique_communities.append(community)
            for idx in community:
                extracted_ids.add(idx)

    return unique_communities

if __name__ == '__main__':
    # Model for computing sentence embeddings. We use one trained for similar questions detection
    model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

    # Load data
    logger.info("Load and concat data")
    scores_dir = Path('/scratch/mt4493/twitter_labor/twitter-labor-data/data/inference/US/DeepPavlov_bert-base-cased-conversational_nov13_iter0_14045091-14114233-evaluation/output/lost_job_1mo')
    random_set_dir = Path('/scratch/mt4493/twitter_labor/twitter-labor-data/data/random_samples/random_samples_splitted/US/evaluation')
    scores_df = pd.concat(
        pd.read_parquet(parquet_file)
        for parquet_file in scores_dir.glob('*.parquet')
    )
    scores_df = scores_df.reset_index()
    logger.info('Loaded scores')
    text_df = pd.concat(
        pd.read_parquet(parquet_file)
        for parquet_file in random_set_dir.glob('*.parquet')
    )
    logger.info('Loaded text')
    df = scores_df.merge(text_df, on="tweet_id", how = 'inner')
    logger.info('Merged both')
    df['rank'] = df['score'].rank(method='dense', ascending=False)
    df = df.sort_values(by=['rank']).reset_index(drop=True)
    max_corpus_size = 50000 # We limit our corpus to only the first 50k lines
    df = df[:max_corpus_size]
    embedding_cache_path = 'top-tweets-{}.pkl'.format(max_corpus_size)
    logger.info("Loaded data. Encode the corpus. This might take a while")

    corpus_sentences = df['text'].tolist()
    corpus_embeddings = model.encode(corpus_sentences, show_progress_bar=True, convert_to_numpy=True)

    # with open(embedding_cache_path, "wb") as fOut:
    #     pickle.dump({'sentences': corpus_sentences, 'embeddings': corpus_embeddings}, fOut)


    logger.info("Start clustering")
    start_time = time.time()

    #Two parameter to tune:
    #min_cluster_size: Only consider cluster that have at least 25 elements (30 similar sentences)
    #threshold: Consider sentence pairs with a cosine-similarity larger than threshold as similar
    clusters = community_detection(corpus_embeddings, min_community_size=25, threshold=0.95)


    #logger.info all cluster / communities
    for i, cluster in enumerate(clusters):
        print("\nCluster {}, #{} Elements ".format(i+1, len(cluster)))
        for sentence_id in cluster:
            print("\t", corpus_sentences[sentence_id])



    logger.info("Clustering done after {:.2f} sec".format(time.time() - start_time))