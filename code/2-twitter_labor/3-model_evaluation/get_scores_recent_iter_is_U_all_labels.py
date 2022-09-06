import pandas as pd
from pathlib import Path
import os



if __name__ == '__main__':
    labels_path = '/scratch/spf248/twitter_labor_market_flows/data/labels/labels.parquet'
    labels_df = pd.read_parquet(labels_path)
    path_data = '/scratch/spf248/scratch_manu/twitter_labor/twitter-labor-data/data/inference/US'
    folder_dict = {9: 'iter_9-convbert-23605411-evaluation',
                   10: 'iter_10-convbert-23672317-evaluation',
                   11: 'iter_11-convbert-23810643-evaluation',
                   12: 'iter_12-convbert-23933846-evaluation',
                   13: 'iter_13-convbert-24263703-evaluation'}
    for iter_nb in [9,10,11,12,13]:
        df = pd.concat([pd.read_parquet(path) for path in
                        Path(os.path.join(path_data, folder_dict[9], 'output', 'is_unemployed')).glob('*.parquet')])
        df = df.reset_index()
        df['tweet_id'] = df['tweet_id'].astype(str)
        df.columns = ['tweet_id', f'score_iter{iter_nb}']
        df[f'rank_iter{iter_nb}'] = df[f'score_iter{iter_nb}'].rank(method='first', ascending=False)
        df = labels_df.merge(df, on=['tweet_id'])
        output_path = f'/scratch/spf248/scratch_manu/twitter_labor/twitter-labor-data/data/active_learning/evaluation_inference/US/all_labels_scored_iter{iter_nb}_is_U.parquet'
        df.to_parquet(output_path, index=False)