import os
import tqdm
import scipy

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

pd.set_option('display.max_columns', None)

data_folder_path = '/Users/dval/work_temp/twitter_from_nyu/output/torch/'

comparator_list = ['laptop', 'nyu']
list_dataframes_to_merge = []
for COMPARATOR in comparator_list:
    data_input_df = pd.DataFrame()
    for file in tqdm.tqdm(os.listdir(data_folder_path)):
        if COMPARATOR in file:

            print('reading', data_folder_path+'/'+file)
            current_file = pd.read_csv(data_folder_path+file)

            if 'onnx_batchsize' not in current_file.columns:
                current_file['batch_size'] = 1
            else:
                current_file['batch_size'] = current_file['onnx_batchsize']

            if 'torch_score' in current_file.columns:
                current_file['score'] = current_file['torch_score']

            if 'torch_time_per_tweet' in current_file.columns:
                current_file['time_per_tweet'] = current_file['torch_time_per_tweet']

            data_input_df = pd.concat([data_input_df,
                                       current_file])
            # break

    # print(data_input_df.head())

    data_input_df = data_input_df[[
           # 'onnx_batchsize', 'onnx_model_type',
           # 'speedup_frac', 'kendalltau', 'spearmanr', 'mean_squared_error',
           'tweet_id',
           'model',
           'score',
           'time_per_tweet',
           'batch_size'
           # 'onnx_time_per_tweet'
        ]]


    data_input_agg_df = data_input_df.groupby(['tweet_id']).agg(['mean', 'std'])
    data_input_agg_df.reset_index(inplace=True)
    # print(data_input_agg_df.head())
    data_input_agg_df.columns = [
                                 'tweet_id',
                                 'score_{}_mean'.format(COMPARATOR),'score_{}_std'.format(COMPARATOR),
                                 'time_per_tweet_{}_mean'.format(COMPARATOR),'time_per_tweet_{}_std'.format(COMPARATOR),
                                 'batch_size','batch_size_std'
                                 # 'batch_size_{}_mean'.format(COMPARATOR),'batch_size_{}_std'.format(COMPARATOR)
                                 # 'onnx_time_per_tweet_mean','onnx_time_per_tweet_std'
                                 ]

    data_input_agg_df = data_input_agg_df[[
        'tweet_id',
        'score_{}_mean'.format(COMPARATOR), 'score_{}_std'.format(COMPARATOR),
        'time_per_tweet_{}_mean'.format(COMPARATOR), 'time_per_tweet_{}_std'.format(COMPARATOR),
        'batch_size'
    ]]

    # print(data_input_agg_df.head())

    list_dataframes_to_merge.append(data_input_agg_df)

from functools import reduce
merged_df = reduce(lambda x,y: pd.merge(x , y,
                            left_on=['tweet_id', 'batch_size'], right_on=['tweet_id', 'batch_size'] , how='inner'),
                           list_dataframes_to_merge)

print(merged_df.shape, list_dataframes_to_merge[0].shape, list_dataframes_to_merge[1].shape)

merged_df['speedup_frac'] = merged_df['time_per_tweet_{}_mean'.format(comparator_list[0])] / merged_df[
    'time_per_tweet_{}_mean'.format(comparator_list[1])]
merged_df['score_rank'] = merged_df['score_{}_mean'.format(comparator_list[0])].rank(method='dense', ascending=False)
merged_df['kendalltau'] = scipy.stats.kendalltau(merged_df['score_rank'],
                                                 merged_df['score_{}_mean'.format(comparator_list[1])]).correlation
merged_df['spearmanr'] = scipy.stats.spearmanr(merged_df['score_{}_mean'.format(comparator_list[0])],
                                               merged_df['score_{}_mean'.format(comparator_list[1])]).correlation
merged_df['mean_squared_error'] = mean_squared_error(merged_df['score_{}_mean'.format(comparator_list[0])],
                                                     merged_df['score_{}_mean'.format(comparator_list[1])])

merged_df['kendalltau'] = -1.0 * merged_df['kendalltau']

print(merged_df.head())

# print(data_input_agg_df.shape)
# print(data_input_agg_df)
# print(data_input_agg_df.columns)



# import pandas as pd
# import rpy2.robjects as ro
# from rpy2.robjects import pandas2ri
# from rpy2.rinterface import parse
# # import rpy2.robjects.lib.ggplot2 as ggplot2
#
# from rpy2.robjects.conversion import localconverter
#
# with localconverter(ro.default_converter + pandas2ri.converter):
#   r_from_pd_df = ro.conversion.py2rpy(data_input_agg_df)
#
# # parse('cat(r_from_pd_df$onnx_batchsize)')
# # test = parse('cat(r_from_pd_df$onnx_batchsize)')
# # print(test[0])
#
# # in the future you might have to use parse https://rpy2.github.io/doc/v3.4.x/html/rinterface.html#parsing-and-evaluating-r-code
# # test = parse(
# # '''
# # plot <- ggplot(data=r_from_pd_df, aes(x=onnx_batchsize, y=speedup_frac_mean)) + geom_point()
# # ggsave(file='test.png', width = 5, height = 5, dpi = 300)
# # '''
# # )
# # print(test)
#
# for Y_AXIS in ['spearmanr', 'mean_squared_error']:
#
#     ro.globalenv['r_output'] = r_from_pd_df
#
#     R_RUN_STRING_TEMPLATE = '''
#     library('ggplot2')
#     r_output$onnx_batchsize <- as.factor(r_output$onnx_batchsize)
#     plot <- ggplot(data=r_output, aes(x=onnx_batchsize, y=**_mean, color=model)) +
#             geom_point(size=0.1)+
#             geom_pointrange(aes(ymin=**_mean-**_std, ymax=**_mean+**_std))+
#             # geom_path()+
#             # geom_jitter()+
#             theme_bw()+
#             theme(legend.position="top")
#     # ggsave(file='plots/**_local.png', width = 5, height = 5, dpi = 300)
#     ggsave(file='plots/**_nyu.png', width = 5, height = 5, dpi = 300)
#     '''
#     R_RUN_STRING = R_RUN_STRING_TEMPLATE.replace('**', Y_AXIS)
#
#
#     print('plotting..\n', R_RUN_STRING)
#     ro.r(R_RUN_STRING)


























