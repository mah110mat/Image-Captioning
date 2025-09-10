import os
import pickle
import random

import numpy as np
import pandas as pd
import json

DATA_PATH = 'Flickr30k_dataset'
IMAGE_PATH = f'{DATA_PATH}/images'

def generate_cache():
    df = pd.read_csv(os.path.join(DATA_PATH, "results.csv"), sep="|")
    #df = pd.read_csv(os.path.join(DATA_PATH, "debug.csv"), sep="|")
    df.columns = [col.strip() for col in df.columns]

    df = df.drop(["comment_number"], axis=1)

    # get every 5 element of the df (5 captions per image) and save image name with corresponding cap tions
    ds = [
        (f'{IMAGE_PATH}/{img_name}', df[df["image_name"] == img_name]["comment"].values)
        for img_name, _ in df[0::5].to_numpy()
    ]

    dataset = {}
    for img_name, array in ds:
        dataset[img_name] = array.tolist()
    num_dataset = len(dataset)
    num_train = int(num_dataset * 0.8)
    num_val = int(num_dataset * 0.1) + num_train

    #import pdb; pdb.set_trace()
    # split 8:1:1 (train/val/test)
    dataset = random.sample(list(dataset.items()), num_dataset)

    with open(os.path.join(DATA_PATH, 'results.json'), 'w') as fp:
        json.dump(dataset, fp)

if __name__ == "__main__":
    if not os.path.isfile( os.path.join(DATA_PATH, 'results.json')):
        generate_cache()

    #import pdb; pdb.set_trace()
    text_data = []
    train_dataset = {}
    print('train_data')
    for (key, val) in dataset[:num_train]:
        va = []
        for vs in val:
            #vs = 'sos' + vs + ' eos'
            try:
                vs = 'sos' + vs + ' eos'
            except:
                print(key, vs)
                import pdb; pdb.set_trace()
            va.append(vs)
            text_data.append(vs)
        train_dataset[key] = va
    with open(os.path.join(DATA_PATH, 'train.json'), 'w') as fp:
        json.dump(train_dataset, fp)

    print('val_data')
    val_dataset = {}
    for (key, val) in dataset[num_train:num_val]:
        va = []
        for vs in val:
            #vs = 'sos' + vs + ' eos'
            try:
                vs = 'sos' + vs + ' eos'
            except:
                print(key, vs)
                import pdb; pdb.set_trace()
            va.append(vs)
            text_data.append(vs)
        val_dataset[key] = va
    with open(os.path.join(DATA_PATH, 'valid.json'), 'w') as fp:
        json.dump(val_dataset, fp)

    print('test_data')
    test_dataset = {}
    for (key, val) in dataset[num_val:]:
        va = []
        for vs in val:
            #vs = 'sos' + vs + ' eos'
            try:
                vs = 'sos' + vs + ' eos'
            except:
                print(key, vs)
                import pdb; pdb.set_trace()
            va.append(vs)
            text_data.append(vs)
        test_dataset[key] = va
    with open(os.path.join(DATA_PATH, 'test.json'), 'w') as fp:
        json.dump(test_dataset, fp)

    with open(os.path.join(DATA_PATH, 'text_data.json'), 'w') as fp:
        json.dump(text_data, fp)


