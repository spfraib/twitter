
# import pandas as pd
# import rpy2
from rpy2.rinterface import evalr
import rpy2.robjects as ro

# from rpy2.robjects import pandas2ri
# from rpy2.robjects.conversion import localconverter
# import rpy2.robjects.lib.ggplot2 as ggplot2
# import rpy2.robjects as ro
# from rpy2.robjects.packages import importr

# with localconverter(ro.default_converter + pandas2ri.converter):
#   r_from_pd_df = ro.conversion.py2rpy(data_input_agg_df)

test = evalr('4.3+9')
# test = rpy2.rinterface.parse('1+5')
print(test[0])
# test = rpy2.rinterface.parse('library(ggplot2)')
# print(test)

