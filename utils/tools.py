import os
import numpy as np
import pandas as pd


def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path


def uniform_scaling(data, max_len):
    seq_len = len(data)
    scaled_data = [data[int(j * seq_len / max_len)] for j in range(max_len)]

    return scaled_data


def save_train_duration(file_name, test_duration):
    res = pd.DataFrame(data=np.zeros((1, 1), dtype=np.float), index=[0],
                       columns=['train_duration'])
    res['train_duration'] = test_duration
    res.to_csv(file_name, index=False)


def save_test_duration(file_name, test_duration):
    res = pd.DataFrame(data=np.zeros((1, 1), dtype=np.float), index=[0],
                       columns=['test_duration'])
    res['test_duration'] = test_duration
    res.to_csv(file_name, index=False)
