import os
import numpy as np
import pandas as pd


def create_directory(directory_path):
    """
    Create a directory if path doesn't exists
    :param directory_path:
    :return:
    """
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path


def save_train_duration(file_name, test_duration):
    """
    Save training time
    :param file_name:
    :param test_duration:
    :return:
    """
    res = pd.DataFrame(data=np.zeros((1, 1), dtype=np.float), index=[0],
                       columns=['train_duration'])
    res['train_duration'] = test_duration
    res.to_csv(file_name, index=False)


def save_test_duration(file_name, test_duration):
    """
    Save test time
    :param file_name:
    :param test_duration:
    :return:
    """
    res = pd.DataFrame(data=np.zeros((1, 1), dtype=np.float), index=[0],
                       columns=['test_duration'])
    res['test_duration'] = test_duration
    res.to_csv(file_name, index=False)
