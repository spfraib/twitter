#gets all this setup
import time
start_time = time.time()

import pandas as pd
import random
import numpy as np
import os

dirs = os.listdir( './' )

counter = 0
for file in dirs:

    if not file.endswith('json'): continue

    results = pd.read_json(file)

    plot_filename = file.split('.')[0]

    results = results.T

    results.index = results['epoch']
    print(results.head())


    results = results[:20]

    print('plotting...', file)

    import matplotlib.pyplot as plt

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
    plt.xticks(np.arange(1, 19, 1.0))
    plt.savefig('plot_{}.png'.format(plot_filename), dpi=100)