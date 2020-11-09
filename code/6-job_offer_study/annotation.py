import csv
import random
import pyarrow.parquet as pq
import pandas as pd

import tokenization
import argparse

# Add arguments
parser = argparse.ArgumentParser("Generate Random tweet set for annotation")
parser.add_argument("--job_offer_file", type = str, help = "File contaning Top Job offer tweets", default = "/scratch/mt4493/twitter_labor/twitter-labor-data/data/inference/DeepPavlov_bert-base-cased-con$
parser.add_argument("--size", type = int, default = 1000, help = "Number of tweets to be selected")
parser.add_argument("--output_file", help = "New_Training_batch.csv")


# Parse Arguments
args = parser.parse_args()
num_tweets = args.size
dataset = args.job_offer_file
output_file = args.output_file


def main():
    
    # Read Random top 350K tweets
    top_tweets = pq.read_table(dataset).to_pandas()

    l = list(range(0,len(top_tweets)))

    # Select num_tweets randomly
    random_tweet_set = random.choices(l, k = num_tweets)

    df = pd.DataFrame(top_tweets,index = random_tweet_set)

    csv_file = open(output_file, mode='w')

    fieldnames = ['Tweet_ID', 'Text','Token', 'ORG', 'LOC', 'JOB_TITLE','Sector']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for id in df.itertuples(index=False):
            index, text = id[0], id[2]
            tweet_tokens = tokenization.tokenize(text)
            for token in tweet_tokens:
                    writer.writerow({'Tweet_ID': index, 'Text': text, 'Token':token, 'ORG':'', 'LOC':'', 'JOB_TITLE':'','Sector':''})


if __name__ == "__main__":
    main()