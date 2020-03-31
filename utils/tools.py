import math
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

from utils.data_processor import uniform_scaling


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


def process_data(regressor_name, X):
    if (regressor_name == "random_forest") or (regressor_name == "xgboost") or (regressor_name == "svr") or ("classic" in regressor_name):
        tmp = []
        for i in tqdm(range(len(X))):
            # 1. flatten
            # 2. fill missing values
            x = X.iloc[i, 0].reset_index(drop=True)
            x.interpolate(method='linear', inplace=True, limit_direction='both')
            tmp2 = list(x)
            for j in range(1, len(X.columns)):
                x = X.iloc[i, j].reset_index(drop=True)
                x.interpolate(method='linear', inplace=True, limit_direction='both')
                tmp2 = tmp2 + list(x)
            tmp2 = pd.DataFrame(tmp2).transpose()

            tmp.append(tmp2)
        X = pd.concat(tmp).reset_index(drop=True)
    else:
        tmp = []
        for i in tqdm(range(len(X))):
            x = X.iloc[i, :]
            _x = x.copy(deep=True)

            # 1. find the maximum length of each dimension
            all_len = [len(y) for y in _x]
            max_len = max(all_len)

            # 2. adjust the length of each dimension
            _y = []
            for y in _x:
                # 2.1 fill missing values
                if y.isnull().any():
                    y = y.interpolate(method='linear', limit_direction='both')

                # 2.2. if length of each dimension is different, uniformly scale the shorted one to the max length
                if len(y) < max_len:
                    y = uniform_scaling(y, max_len)
                _y.append(y)

            tmp.append(np.array(np.transpose(_y)))
        X = np.array(tmp)
    return X


def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


def calculate_regression_metrics(y_true, y_pred, duration=None, y_true_val=None, y_pred_val=None):
    res = pd.DataFrame(data=np.zeros((1, 3), dtype=np.float), index=[0],
                       columns=['rmse', 'mae', 'duration'])
    res['rmse'] = math.sqrt(mean_squared_error(y_true, y_pred))
    res['mae'] = mean_absolute_error(y_true, y_pred)

    if not y_true_val is None:
        # this is useful when transfer learning is used with cross validation
        res['rmse_val'] = math.sqrt(mean_squared_error(y_true_val, y_pred_val))
        res['mae_val'] = mean_absolute_error(y_true_val, y_pred_val)
    if duration is not None:
        res['duration'] = duration
    return res


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


def save_logs_for_regression(output_directory, y_pred, y_true, duration, y_true_val=None,
                             y_pred_val=None):
    df_metrics = calculate_regression_metrics(y_true, y_pred, duration, y_true_val, y_pred_val)
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

    return df_metrics


def save_logs_for_regression_deep_learning(output_directory, hist, lr=True):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory + 'history.csv', index=False)

    index_best_model = hist_df['loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_mae',
                                          'best_model_val_mae', 'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['mae']
    df_best_model['best_model_val_acc'] = row_best_model['val_mae']
    if lr == True:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)

    # for FCN there is no hyperparameters fine tuning - everything is static in code

    # plot losses
    plot_epochs_metric(hist, output_directory + 'epochs_loss.png')
