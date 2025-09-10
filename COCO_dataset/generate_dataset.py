import os
import pickle
import random

import numpy as np
import pandas as pd
import json

#from dataset import valid_test_split
def valid_test_split(captions_mapping_valid, NUM_VALID_IMG):
    valid_data={}
    test_data={}
    conta_valid = 0
    for id in captions_mapping_valid:
        if conta_valid<NUM_VALID_IMG:
            valid_data.update({id : captions_mapping_valid[id]})
            conta_valid+=1
        else:
            test_data.update({id : captions_mapping_valid[id]})
            conta_valid+=1
    return valid_data, test_data

with open("captions_mapping_valid.json", 'r') as json_file:
    valid_data = json.load(json_file)

valid_data, test_data  = valid_test_split(valid_data, 30000)
print(len(valid_data))
print(len(test_data))

with open('valid.json', 'w') as json_file:
    json.dump(valid_data, json_file)
with open('test.json', 'w') as json_file:
    json.dump(test_data,json_file)