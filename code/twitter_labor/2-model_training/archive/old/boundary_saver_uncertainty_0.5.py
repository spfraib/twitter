import os

workers = os.cpu_count()
if 'sched_getaffinity' in dir(os):
    workers = len(os.sched_getaffinity(0))
print('number of cpus:', workers)

import re
with open('/proc/meminfo') as f:
    meminfo = f.read()
matched = re.search(r'^MemTotal:\s+(\d+)', meminfo)
if matched:
    mem_total_kB = int(matched.groups()[0])

print('memory available (GB):', mem_total_kB / 1024 / 1024)


#
# # filtered contains 8G of data!!
# import time
# start_time = time.time()
# import pyarrow.parquet as pq
# from glob import glob
# import os
# country_code = 'US'
# month = '2012-1'
# path_to_data = '/scratch/spf248/twitter/data/classification/US/filtered/'
# tweets_filtered=pq.ParquetDataset(glob(os.path.join(path_to_data,
# #                                            country_code,
# #                                            month,
#                                            '*.parquet'))).read().to_pandas()
# print('time taken to load keyword filtered sample:', str(time.time() - start_time), 'seconds')
# print(tweets_filtered.shape)
#
#
# # random contains 7.3G of data!!
# import time
# start_time = time.time()
# import pyarrow.parquet as pq
# from glob import glob
# import os
# country_code = 'US'
# month = '2012-1'
# path_to_data = '/scratch/spf248/twitter/data/classification/US/random'
# tweets_random=pq.ParquetDataset(glob(os.path.join(path_to_data,
# #                                            country_code,
# #                                            month,
#                                            '*.parquet'))).read().to_pandas()
# print('time taken to load random sample:', str(time.time() - start_time), 'seconds')
# print(tweets_random.shape)
#
#
#
#
# import glob
# import pandas as pd
# import time
# start_time = time.time()
# model_output_path = '/scratch/spf248/twitter/data/classification/US/BERT/twitter_sam/mturk_mar6/pred/'
# model_output_filtered = pd.concat([pd.read_csv(f) for f in glob.glob(model_output_path+'filtered*.csv')], ignore_index = True)
# print('time taken to load filtered sample:', str(time.time() - start_time), 'seconds')
#
# import glob
# import pandas as pd
# import time
# start_time = time.time()
# model_output_path = '/scratch/spf248/twitter/data/classification/US/BERT/twitter_sam/mturk_mar6/pred/'
# model_output_random = pd.concat([pd.read_csv(f) for f in glob.glob(model_output_path+'random*.csv')], ignore_index = True)
# print('time taken to load random sample:', str(time.time() - start_time), 'seconds')
#
# model_output_random.columns = ['tweet_id', 'offer_model', 'search_model', 'unemployed_model', 'hired_model', 'loss_model']
# model_output_filtered.columns = ['tweet_id', 'search_model', 'unemployed_model', 'offer_model', 'hired_model', 'loss_model']
#
#
# start_time = time.time()
# tweets_filtered['tweet_id'] = tweets_filtered['tweet_id'].apply(pd.to_numeric)
# print('time taken for filtered conversion:', str(time.time() - start_time), 'seconds')
#
# start_time = time.time()
# tweets_random['tweet_id'] = tweets_random['tweet_id'].apply(pd.to_numeric)
# print('time taken for random conversion:', str(time.time() - start_time), 'seconds')
#
#
#
# start_time = time.time()
# merged_filtered = tweets_filtered.merge(model_output_filtered, on='tweet_id')
# print(merged_filtered.shape)
# print('time taken for merge filtered:', str(time.time() - start_time), 'seconds')
#
# start_time = time.time()
# merged_random = tweets_random.merge(model_output_random, on='tweet_id')
# print(merged_random.shape)
# print('time taken for merge random:', str(time.time() - start_time), 'seconds')

import pandas as pd
start_time = time.time()
# merged_filtered.to_pickle('../mturk_mar6/boundary/merged_filtered.pickle')
merged_filtered =
print('time taken for pickle filtered:', str(time.time() - start_time), 'seconds')

start_time = time.time()
# merged_random.to_pickle('../mturk_mar6/boundary/merged_random.pickle')
print('time taken for pickle random:', str(time.time() - start_time), 'seconds')



boundary = 0.5
topN = 100
columns = ['offer_model','search_model','unemployed_model','hired_model','loss_model']

import matplotlib.pyplot as plt

for column in columns:
    # all_filtered_boundary = merged_filtered.loc[(merged_filtered[column] >= threshold - boundary_width) &
                                                # (merged_filtered[column] <= threshold + boundary_width)]
    boundary_data = merged_filtered
    boundary_data['dist_point5'] = abs(merged_filtered[column] - boundary)

    start_time = time.time()
    print(column, 'subtraction time taken:', str(time.time() - start_time), 'seconds')

    start_time = time.time()
    boundary_data = boundary_data.sort_values(by=['dist_point5'], ascending=True)
    print(column, 'sorting time taken:', str(time.time() - start_time), 'seconds')

    start_time = time.time()
    all_filtered_boundary[:topN].to_csv('../mturk_mar6/boundary/filtered_{}.csv'.format(column))
    # print(all_filtered_boundary['text'])
    print(column, 'write csv time taken:', str(time.time() - start_time), 'seconds')

    start_time = time.time()
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.hist(merged_filtered[column], bins=10, density = True)
    ax.set_yscale('log')
    # plt..yscale(value)
    plt.title(column)
    ax.legend()
    #plt.show()
    plt.savefig('../mturk_mar6/boundary/filtered_{}_log.png'.format(column))

    print(column, 'plot filtered time taken:', str(time.time() - start_time), 'seconds')


