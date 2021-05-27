import json
import os
from collections import defaultdict

path = '/home/manuto/Documents/world_bank/bert_twitter_labor/twitter-labor-data/data/evaluation_metrics/US/diversity/threshold_calibrated_distance_with_seed'
json_list = os.listdir(path)
final_dict = defaultdict(set)

for json_file in json_list:
    with open(os.path.join(path, json_file), 'r') as JSON:
        json_dict = json.load(JSON)
    for k, v in json_dict.items():  # d.items() in Python 3+
        if k not in final_dict.keys():
            final_dict[k] = dict()
        for k_2, v_2 in json_dict[k].items():
            if k_2 not in final_dict[k].keys():
                final_dict[k][k_2] = dict()
            for k_3, v_3 in json_dict[k][k_2].items():
                if k_3 not in final_dict[k][k_2].keys():
                    final_dict[k][k_2][k_3] = json_dict[k][k_2][k_3]

print(final_dict['our_method'])
print('\n')
print(final_dict['adaptive'])
print('\n')
print(final_dict['uncertainty'])