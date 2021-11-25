from pathlib import Path
import argparse
import pandas as pd

logging.basicConfig(
                    # filename=f'{args.log_path}.log',
                    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args_from_command_line()
    logger.info(args.country_code)
    path_to_tweets = os.path.join(
        '/scratch/spf248/twitter/data/user_timeline/user_timeline_text_preprocessed', args.country_code)
    path_to_samples = os.path.join(
        '/scratch/spf248/twitter/data/user_timeline/user_timeline_evaluation_samples', args.country_code)
    for path in Path(path_to_tweets).glob('*.parquet'):
        tweets_df = pd.read_parquet(path, columns=['tweet_id', 'text'])
        logger.info(path)
        tweets_df = tweets_df.loc[tweets_df['tweet_id'].isin(list(scores_df['tweet_id'].unique()))]
        if tweets_df.shape[0] > 0:
            final_df_list.append(tweets_df)
    logger.info('Finished retrieving tweets with indices.')
    tweets_df = pd.concat(final_df_list).reset_index(drop=True)
    df = tweets_df.merge(scores_df, on=['tweet_id']).reset_index(drop=True)
    df = df[['tweet_id', 'text', label, 'rank']]
    df.columns = ['tweet_id', 'text', 'score', 'rank']
    df = df.sort_values(by=['score'], ascending=False).reset_index(drop=True)
    output_path = os.path.join(path_to_evals, f'{label}.parquet')
    df.to_parquet(output_path, index=False)