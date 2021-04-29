import os
import tqdm
import scipy

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

pd.set_option('display.max_columns', None)

# data_folder_path = '/Users/dval/work_temp/twitter_from_nyu/output_local'
data_folder_path = '/Users/dval/work_temp/twitter_from_nyu/nyu_output_speedtest_standalone'

torch_reference_df = pd.read_csv(data_folder_path+'/torch_reference.csv')
torch_reference_df = torch_reference_df[['tweet_id',
                                         'torch_score',
                                         'torch_time_per_tweet'
                                         ]]

# print(torch_reference_df.head())

data_input_df = pd.DataFrame()

for file in tqdm.tqdm(os.listdir(data_folder_path+'/job_search/')):
    # print('reading', data_folder_path+'/'+file)
    current_file = pd.read_csv(data_folder_path+'/job_search/'+file)
    # print(current_file.dtypes)
    # print(current_file.head())
    #since the columns we care about are the same for all rows

    merged = current_file.merge(torch_reference_df)
    # print(merged.head())
    # merged = onnx_predictions_random_df.merge(tweets_random)
    # merged = merged.merge(torch_predictions_random_df)

    # merged['speedup_frac'] = (torch_per_tweet)/onnx_per_tweet

    merged['speedup_frac'] = merged['torch_time_per_tweet']/merged['onnx_time_per_tweet']
    merged['pytorch_score_rank'] = merged['torch_score'].rank(method='dense', ascending=False)
    merged['kendalltau'] = scipy.stats.kendalltau(merged['pytorch_score_rank'], merged['onnx_score']).correlation
    merged['spearmanr'] = scipy.stats.spearmanr(merged['torch_score'], merged['onnx_score']).correlation
    merged['mean_squared_error'] = mean_squared_error(merged['torch_score'], merged['onnx_score'])

    data_input_df = pd.concat([data_input_df,
                               merged])
    # break

data_input_df = data_input_df[[
       'onnx_batchsize', 'onnx_model_type',
       'speedup_frac', 'kendalltau', 'spearmanr', 'mean_squared_error',
       'torch_time_per_tweet', 'onnx_time_per_tweet'
    ]]

data_input_agg_df = data_input_df.groupby(['onnx_batchsize', 'onnx_model_type']).agg(['mean', 'std'])
data_input_agg_df.reset_index(inplace=True)
print(data_input_agg_df.head())
data_input_agg_df.columns = ['onnx_batchsize', 'onnx_model_type',
                             'speedup_frac_mean','speedup_frac_std',
                             'kendalltau_mean','kendalltau_std',
                             'spearmanr_mean','spearmanr_std',
                             'mean_squared_error_mean','mean_squared_error_std',
                             'torch_time_per_tweet_mean','torch_time_per_tweet_std',
                             'onnx_time_per_tweet_mean','onnx_time_per_tweet_std'
                             ]

data_input_agg_df['model'] = data_input_agg_df['onnx_model_type'].astype(str).str[:-5]
data_input_agg_df['kendalltau_mean'] = -1.0 * data_input_agg_df['kendalltau_mean']


# print(data_input_agg_df.shape)
# print(data_input_agg_df)
# print(data_input_agg_df.columns)



import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.rinterface import parse
# import rpy2.robjects.lib.ggplot2 as ggplot2

from rpy2.robjects.conversion import localconverter

with localconverter(ro.default_converter + pandas2ri.converter):
  r_from_pd_df = ro.conversion.py2rpy(data_input_agg_df)

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

for Y_AXIS in ['onnx_time_per_tweet', 'torch_time_per_tweet', 'speedup_frac', 'kendalltau', 'spearmanr', 'mean_squared_error']:
# rdf = pandas2ri.py2ri(all_results)
# Y_AXIS = 'onnx_time_per_tweet'
# Y_AXIS = 'torch_time_per_tweet'
# Y_AXIS = 'speedup_frac'
# Y_AXIS = 'kendalltau'
# Y_AXIS = 'spearmanr'
# Y_AXIS = 'mean_squared_error'

    ro.globalenv['r_output'] = r_from_pd_df

    R_RUN_STRING_TEMPLATE = '''
    library('ggplot2')
    r_output$onnx_batchsize <- as.factor(r_output$onnx_batchsize)
    plot <- ggplot(data=r_output, aes(x=onnx_batchsize, y=**_mean, color=model)) + 
            geom_point(size=0.1)+
            geom_pointrange(aes(ymin=**_mean-**_std, ymax=**_mean+**_std))+
            # geom_path()+
            # geom_jitter()+
            theme_bw()+
            theme(legend.position="top")            
    # ggsave(file='plots/**_local.png', width = 5, height = 5, dpi = 300)
    ggsave(file='plots/**_nyu.png', width = 5, height = 5, dpi = 300)
    '''
    R_RUN_STRING = R_RUN_STRING_TEMPLATE.replace('**', Y_AXIS)


    print('plotting..\n', R_RUN_STRING)
    ro.r(R_RUN_STRING)

# geom_point()+
# geom_pointrange(aes(ymin=estimate-CI, ymax=estimate+CI), alpha=0.9, fill = 'darkgreen')+
# geom_hline(yintercept=0, color = 'red', alpha = 0.3) +
# geom_vline(xintercept=0, color = 'red', alpha = 0.3) +
# xlab("Days after lifting mandate") +
# ylab("Estimated Change in Mask Adherance
# (percentage points)")+
# theme_bw()+
# theme(legend.position = "none")+
# ggtitle("Effect of Lifting Mask Mandates on Mask Adherence")+
# theme(plot.title = element_text(hjust = 0.5))+
# scale_x_continuous(breaks = seq(-40, 50, by = 10))



























