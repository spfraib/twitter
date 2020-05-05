#gets all this setup
import time
start_time = time.time()

import pandas as pd
import random
import numpy as np

results = pd.read_json('./mturk_mar6/results.json')

results = results.T

results.index = results['epoch']
print(results.head())


# results = results[:20]

print('plotting...')

# # from plotnine import ggplot, geom_point, aes, stat_smooth
# from plotnine import ggplot, geom_point, aes, stat_smooth
# # from plotnine.data import mtcars
#
# my_plot = (ggplot(results)
#  + geom_point(aes('epoch', 'train_loss'), size = .1, alpha = 0.3, color='red')
#  + geom_point(aes('epoch', 'loss'), size = .1, alpha = 0.3, color='red')
#  # + stat_smooth(size = .8, alpha = 0.5, se = False, method = 'loess')
#            )

 # + facet_wrap('~gear'))

# print(p)
# my_plot.save("results.png", width=5, height=5, dpi=300)


# import numpy as np
import matplotlib.pyplot as plt

# # Create some mock data
# t = np.arange(0.01, 10.0, 0.01)
# data1 = np.exp(t)
# data2 = np.sin(2 * np.pi * t)

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('epoch')
ax1.set_ylabel('train_loss', color=color)
ax1.plot(results['epoch'], results['train_loss'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('val loss', color=color)  # we already handled the x-label with ax1
ax2.plot(results['epoch'], results['loss'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()

plt.savefig('results_20.png', dpi=100)