import getpass

import pandas as pd

from utils.data_loader import load_from_tsfile_to_dataframe
from utils.tools import create_directory, process_data, calculate_regression_metrics

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def fit_regressor(output_directory, regressor_name, X_train, y_train, X_test, y_test):
    input_shape = X_train.shape[1:]

    regressor = create_regressor(regressor_name, input_shape, output_directory)
    regressor.fit(X_train, y_train, X_test, y_test)

    return regressor


def create_regressor(regressor_name, input_shape, output_directory, verbose=1):
    if regressor_name == "resnet":
        from regressors.resnet import ResNetRegressor
        return ResNetRegressor(output_directory, input_shape, verbose)
    if regressor_name == "fcn":
        from regressors.fcn import FCNRegressor
        return FCNRegressor(output_directory, input_shape, verbose)
    if regressor_name == "random_forest":
        from regressors.random_forest import RFRegressor
        return RFRegressor(output_directory)
    if regressor_name == "xgboost":
        from regressors.xgboost import XGBoostRegressor
        return XGBoostRegressor(output_directory)


username = getpass.getuser()
print('Username: {}'.format(username))
regressor_name = "random_forest"  # resnet, fcn, xgboost, random_forest

problems = ["PPGDalia", "EthanolConcentration", "AcetoneConcentration", "BenzeneConcentration",
            "NewsHeadlineSentiment", "NewsTitleSentiment",
            "IEEEPPG", "CoolerCondition", "ValveCondition", "PumpLeakage", "HydraulicAccumulator"]
problem = "Sample"

data_folder = "data/" + problem + "/"
print("[Experiments] Data folder: {}".format(data_folder))

train_file = data_folder + problem + "_TRAIN.ts"
test_file = data_folder + problem + "_TEST.ts"

print("[Experiments] Loading train data")
X_train, y_train = load_from_tsfile_to_dataframe(train_file)

print("[Experiments] Loading test data")
X_test, y_test = load_from_tsfile_to_dataframe(test_file)

print("[Experiments] Reshaping data")
X_train = process_data(regressor_name, X_train)
X_test = process_data(regressor_name, X_test)

print("[Experiments] X_train: {}".format(X_train.shape))
print("[Experiments] X_test: {}".format(X_test.shape))

input_shape = X_train.shape[1:]
verbose = True

itr = "iter_0"
output_directory = "demo_output/"
output_directory = output_directory + regressor_name + '/' + problem + '/' + itr + '/'
create_directory(output_directory)

model = fit_regressor(output_directory, regressor_name, X_train, y_train, X_test, y_test)

y_pred, test_duration = model.predict(X_test)
df_metrics = calculate_regression_metrics(y_test, y_pred, test_duration)
df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

print("[Experiments]", y_pred)
print("[Experiments]", y_test)
print("[Experiments]", df_metrics)
