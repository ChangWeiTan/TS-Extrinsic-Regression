import math
from time import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm

from utils.tools import uniform_scaling

name = "RegressorTools"

classical_ml_models = ["xgboost", "svr", "random_forest"]
deep_learning_models = ["fcn", "resnet", "inception"]
tsc_models = ["rocket"]


def fit_regressor(output_directory, regressor_name, X_train, y_train,
                  X_val=None, y_val=None, itr=1):
    print("[{}] Fitting regressor".format(name))
    start_time = time()

    input_shape = X_train.shape[1:]

    regressor = create_regressor(regressor_name, input_shape, output_directory, itr)
    if (X_val is not None) and (regressor_name in deep_learning_models):
        regressor.fit(X_train, y_train, X_val, y_val)
    else:
        regressor.fit(X_train, y_train)
    elapsed_time = time() - start_time
    print("[{}] Classifier fitted, took {}s".format(name, elapsed_time))
    return regressor


def create_regressor(regressor_name, input_shape, output_directory, verbose=1, itr=1):
    print("[{}] Creating regressor".format(name))
    # SOTA TSC deep learning
    if regressor_name == "resnet":
        from models.deep_learning import resnet
        return resnet.ResNetRegressor(output_directory, input_shape, verbose)
    if regressor_name == "fcn":
        from models.deep_learning import fcn
        return fcn.FCNRegressor(output_directory, input_shape, verbose)
    if regressor_name == "inception":
        from models.deep_learning import inception
        return inception.InceptionTimeRegressor(output_directory, input_shape, verbose)

    if regressor_name == "rocket":
        from models import rocket
        return rocket.RocketRegressor(output_directory, verbose)

    # classical ML models
    if regressor_name == "xgboost":
        from models import classical_models
        kwargs = {"n_estimators": 100,
                  "n_jobs": 0,
                  "learning_rate": 0.1,
                  "random_state": itr - 1,
                  "verbosity  ": verbose}
        return classical_models.XGBoostRegressor(output_directory, verbose, kwargs)
    if regressor_name == "random_forest":
        from models import classical_models
        kwargs = {"n_estimators": 100,
                  "n_jobs": -1,
                  "random_state": itr - 1,
                  "verbose": verbose}
        return classical_models.RFRegressor(output_directory, verbose, kwargs)
    if regressor_name == "svr":
        from models import classical_models
        return classical_models.SVRRegressor(output_directory, verbose)


def process_data(regressor_name, X, min_len, normalise=None):
    if regressor_name in classical_ml_models:
        tmp = []
        for i in tqdm(range(len(X))):
            # 1. flatten
            # 2. fill missing values

            tmp2 = []
            for j in range(0, len(X.columns)):
                x = X.iloc[i, j].reset_index(drop=True)
                x.interpolate(method='linear', inplace=True, limit_direction='both')
                x = x.values[:min_len]
                if normalise == "standard":
                    x = StandardScaler().fit_transform(x.reshape(-1, 1))
                elif normalise == "minmax":
                    x = MinMaxScaler().fit_transform(x.reshape(-1, 1))
                x = pd.DataFrame(x)
                tmp2 = tmp2 + x.values.tolist()
            tmp2 = pd.DataFrame(tmp2).transpose()

            tmp.append(tmp2)

        X = pd.concat(tmp).reset_index(drop=True)
    else:
        tmp = []
        for i in tqdm(range(len(X))):
            _x = X.iloc[i, :].copy(deep=True)

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
            _y = np.array(np.transpose(_y))

            # 3. adjust the length of the series, chop of the longer series
            _y = _y[:min_len, :]

            # 4. normalise the series
            if normalise == "standard":
                scaler = StandardScaler().fit(_y)
                _y = scaler.transform(_y)
            if normalise == "minmax":
                scaler = MinMaxScaler().fit(_y)
                _y = scaler.transform(_y)

            tmp.append(_y)
        X = np.array(tmp)
    return X


def calculate_regression_metrics(y_true, y_pred, y_true_val=None, y_pred_val=None):
    res = pd.DataFrame(data=np.zeros((1, 2), dtype=np.float), index=[0],
                       columns=['rmse', 'mae'])
    res['rmse'] = math.sqrt(mean_squared_error(y_true, y_pred))
    res['mae'] = mean_absolute_error(y_true, y_pred)

    if not y_true_val is None:
        # this is useful when transfer learning is used with cross validation
        res['rmse_val'] = math.sqrt(mean_squared_error(y_true_val, y_pred_val))
        res['mae_val'] = mean_absolute_error(y_true_val, y_pred_val)

    return res
