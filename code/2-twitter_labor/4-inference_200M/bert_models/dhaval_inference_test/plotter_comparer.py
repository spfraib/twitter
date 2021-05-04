import os
import tqdm
import scipy

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

pd.set_option('display.max_columns', None)

data_folder_path = '/Users/dval/work_temp/twitter_from_nyu/output/replication_output_data/'

"""
Note:
1. /Users/dval/work_temp/twitter_from_nyu/output/replication_output_data/mt4493_random-converted-optimized-quantized
.onnx-*.csv contains the output from onnx_baselines.py (where inference is done in batches of different sizes and 
with different models) run on NYU server

2. /Users/dval/work_temp/twitter_from_nyu/output/replication_output_data/random-laptop_converted-optimized.onnx-bs
-*.csv: same as above: output produced by onnx_baselines.py but run on my laptop

3. /Users/dval/work_temp/twitter_from_nyu/output/replication_output_data/mt4493_current_manu_inference_no_BATCHES
-manu_current-bs-*.csv is 
produced with dhaval_inference_ONNX_bert_100M_random.py which is the current/Manuel way of running inference (run on 
NYU)

4. /Users/dval/work_temp/twitter_from_nyu/output/replication_output_data/torch_reference_rep-*_torch_laptop.csv: 
produced by 
torch_baseline.py run on laptop 

5. /Users/dval/work_temp/twitter_from_nyu/output/replication_output_data/torch_reference_rep-*_torch_nyu.csv same as 
above, run on NYU 

"""
comparator_list = [
    {'manu_inference_no_BATCHES': 'statusquo'}, # manu current approach no batches, onnx, see 3.
    {'torch_laptop': 'laptop'}, #torch on laptop see 4. above
    # {'torch_nyu': 'nyu'}, #torch on laptop see 4. above
    {'random-laptop_converted': 'laptop'}
                   ]
assert len(comparator_list) == 2, 'you can only compare two runs at a time yo'

def is_unique(s):
    a = s.to_numpy() # s.values (pandas<0.24)
    return (a[0] == a).all()

model_list = []
list_dataframes_to_merge = []
for COMPARATOR_dict in comparator_list:
    file_string = list(COMPARATOR_dict.items())[0][0]
    location = list(COMPARATOR_dict.items())[0][1]
    data_input_df = pd.DataFrame()
    for file in tqdm.tqdm(os.listdir(data_folder_path)):
        if file_string in file:

            print('reading', data_folder_path+'/'+file)
            current_file = pd.read_csv(data_folder_path+file)

            if 'onnx_batchsize' not in current_file.columns:
                current_file['batch_size'] = 1
            else:
                current_file['batch_size'] = current_file['onnx_batchsize']

            if 'torch_score' in current_file.columns:
                current_file['score'] = current_file['torch_score']

            if 'onnx_score' in current_file.columns:
                current_file['score'] = current_file['onnx_score']

            if 'torch_time_per_tweet' in current_file.columns:
                current_file['time_per_tweet'] = current_file['torch_time_per_tweet']

            if 'onnx_time_per_tweet' in current_file.columns:
                current_file['time_per_tweet'] = current_file['onnx_time_per_tweet']

            if 'onnx_model_type' in current_file.columns:
                current_file['model'] = current_file['onnx_model_type']

            data_input_df = pd.concat([data_input_df,
                                       current_file])
            # break

    # print(data_input_df.head())

    # check that all models in the column are the same
    assert is_unique(data_input_df['model'])
    model = data_input_df['model'].iloc[0]
    model_list.append(model)

    data_input_df = data_input_df[[
           # 'onnx_batchsize', 'onnx_model_type',
           # 'speedup_frac', 'kendalltau', 'spearmanr', 'mean_squared_error',
           'tweet_id',
           # 'model',
           'score',
           'time_per_tweet',
           'batch_size'
           # 'onnx_time_per_tweet'
        ]]


    # data_input_agg_df = data_input_df.groupby(['tweet_id', 'model', 'batch_size']).agg(['mean', 'std'])
    data_input_agg_df = data_input_df.groupby(['tweet_id', 'batch_size']).agg(['mean', 'std'])
    data_input_agg_df.reset_index(inplace=True)
    # print(data_input_agg_df.head())
    # modifier_string = location + "_" + model
    modifier_string = location + "_" + model
    data_input_agg_df.columns = [
                                 'tweet_id',
                                 # 'model',
                                 'batch_size',
                                 'score_{}'.format(modifier_string),
                                 'score_{}_std'.format(modifier_string),
                                 'time_per_tweet_{}'.format(modifier_string),
                                 'time_per_tweet_{}_std'.format(modifier_string),
                                 # 'batch_size','batch_size_std'
                                 # 'batch_size_{}_mean'.format(COMPARATOR),'batch_size_{}_std'.format(COMPARATOR)
                                 # 'onnx_time_per_tweet_mean','onnx_time_per_tweet_std'
                                 ]

    data_input_agg_df = data_input_agg_df[[
        'tweet_id',
        # 'model',
        'batch_size',
        'score_{}'.format(modifier_string),
        # 'score_{}_std'.format(modifier_string),
        'time_per_tweet_{}'.format(modifier_string),
        # 'time_per_tweet_{}_std'.format(modifier_string),
    ]]


    list_dataframes_to_merge.append(data_input_agg_df)

from functools import reduce
merged_df = reduce(lambda x,y: pd.merge(x , y,
                            # left_on=['tweet_id', 'batch_size', 'model'],
                            left_on=['tweet_id', 'batch_size'],
                            # right_on=['tweet_id', 'batch_size', 'model'],
                            right_on=['tweet_id', 'batch_size'],
                            how='inner'),
                   list_dataframes_to_merge)

print(merged_df.shape, list_dataframes_to_merge[0].shape, list_dataframes_to_merge[1].shape)
print(merged_df.head())


first_comparator = list(comparator_list[0].items())[0][1] + "_" + model_list[0]
second_comparator = list(comparator_list[1].items())[0][1] + "_" + model_list[1]
# second_comparator = list(comparator_list[1].items())[0][1] + "_" + list(comparator_list[1].items())[0][0]

merged_df['speedup_frac'] = merged_df['time_per_tweet_{}'.format(first_comparator)] / merged_df[
    'time_per_tweet_{}'.format(second_comparator)]

merged_df['score_rank'] = merged_df['score_{}'.format(first_comparator)].rank(method='dense', ascending=False)
merged_df['kendalltau'] = scipy.stats.kendalltau(merged_df['score_rank'],
                                                 merged_df['score_{}'.format(second_comparator)]).correlation
merged_df['spearmanr'] = scipy.stats.spearmanr(merged_df['score_{}'.format(first_comparator)],
                                               merged_df['score_{}'.format(second_comparator)]).correlation
merged_df['MSE'] = mean_squared_error(merged_df['score_{}'.format(first_comparator)],
                                                     merged_df['score_{}'.format(second_comparator)])

merged_df['kendalltau'] = -1.0 * merged_df['kendalltau']

# print(merged_df.head())

r_df = merged_df[[
    # 'model',
    'batch_size',
    'spearmanr',
    'MSE',
    'kendalltau'
]]

# print(r_df.head())

r_simplified_df = r_df.drop_duplicates()
print(r_df.shape, r_simplified_df)

print(r_simplified_df.head())


import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.rinterface import parse
# import rpy2.robjects.lib.ggplot2 as ggplot2

from rpy2.robjects.conversion import localconverter

with localconverter(ro.default_converter + pandas2ri.converter):
  r_from_pd_df = ro.conversion.py2rpy(r_simplified_df)

# parse('cat(r_from_pd_df$onnx_batchsize)')
# test = parse('cat(r_from_pd_df$onnx_batchsize)')
# print(test[0])

# in the future you might have to use parse https://rpy2.github.io/doc/v3.4.x/html/rinterface.html#parsing-and-evaluating-r-code
# test = parse(
# '''
# plot <- ggplot(data=r_from_pd_df, aes(x=onnx_batchsize, y=speedup_frac_mean)) + geom_point()
# ggsave(file='test.png', width = 5, height = 5, dpi = 300)
# '''
# )
# print(test)

# filename_endstring = comparator_list[0]+"_vs_"+comparator_list[1]
filename_endstring = first_comparator + "_vs_" + second_comparator

for Y_AXIS in ['spearmanr', 'MSE', 'kendalltau']:

    print('>>plotting:', filename_endstring, Y_AXIS)

    ro.globalenv['r_output'] = r_from_pd_df

    R_RUN_STRING_TEMPLATE = '''
    library('ggplot2')
    # r_output$onnx_batchsize <- as.factor(r_output$onnx_batchsize)
    plot <- ggplot(data=r_output, aes(x=batch_size, y=**)) +
    # plot <- ggplot(data=r_output, aes(x=batch_size, y=**, color=model)) +
            geom_point(size=0.1)+
            # geom_pointrange(aes(ymin=**_mean-**_std, ymax=**_mean+**_std))+
            # geom_path()+
            # geom_jitter()+
            theme_bw()+
            theme(legend.position="top")
    ggsave(file='plots/temp/**_REPLACE2.png', width = 5, height = 5, dpi = 300)
    '''
    R_RUN_STRING = R_RUN_STRING_TEMPLATE.replace('**', Y_AXIS)
    R_RUN_STRING = R_RUN_STRING.replace('REPLACE2', filename_endstring)


    print('plotting..\n', R_RUN_STRING)
    ro.r(R_RUN_STRING)


























