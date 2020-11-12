import urllib
from urllib.request import urlopen
import http.client
import pandas as pd
import re
import requests

def is_url_down(url):
    try:
        code = urlopen(url).getcode()
        if code == 200:
            print(code)
            return True
    except Exception as e:
        print(e)
        return False


# def unshorten_url(url):
#     response = requests.get(url)
#     if response.status_code == 301:
#         return unshorten_url(response.redirect_destination)
#     elif response.status_code == 200:
#         if response.body.contains_meta_redirect:
#             return unshorten_url(response.body.meta_redirect_destination)
#         else:
#             return url
#     else:
#         print('Exception')


def main():
    df = pd.read_pickle('/home/manuto/Documents/world_bank/bert_twitter_labor/data/sample_top_tweets/convBERT/top_tweets_random_job_offer_it0_convBERT.pkl')
    for i in range(df.shape[0]):
        link_str = re.search("(?P<url>https?://[^\s]+)", df['text'][i]).group("url")
        print(link_str)
        is_url_down(link_str)

if __name__ == '__main__':
    main()



