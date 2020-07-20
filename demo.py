import getopt
import sys

import numpy as np
import pandas as pd

from utils import data_loader
from utils.data_loader import load_from_tsfile_to_dataframe
from utils.regressor_tools import process_data, fit_regressor, calculate_regression_metrics
from utils.tools import create_directory

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

module = "RegressionExperiment"
task = "regression"
itr = 1
problem = "Covid3Month"
regressor_name = "rocket"
norm = "none"
machine = "linux"

try:
    opts, args = getopt.getopt(sys.argv[1:], "hp:r:i:n:m:", ["problem=", "regressor=", "iter=", "norm=", "machine="])
except getopt.GetoptError:
    print("demo.py -p <problem> -r <regressor> -i <iteration> -n <normalisation> -m <machine>")
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print("demo.py -p <problem> -s <regressor> -i <iteration> -n <normalisation> -m <machine>")
        sys.exit()
    elif opt in ("-p", "--problem"):
        problem = arg
    elif opt in ("-r", "--regressor"):
        regressor_name = arg
    elif opt in ("-i", "--iter"):
        itr = arg
    elif opt in ("-n", "--norm"):
        norm = arg
    elif opt in ("-m", "--machine"):
        machine = arg

data_path = data_loader.get_data_path(task=task, machine=machine)
output_directory = data_loader.get_output_path(task=task, machine=machine)

if norm != "none":
    output_directory = output_directory.replace(task + "/", task + "_" + norm + "/")
output_directory = output_directory + regressor_name + '/' + problem + '/itr_' + str(itr) + '/'
create_directory(output_directory)

if __name__ == '__main__':
    print("=======================================================================")
    print("[{}] Starting Holdout Experiments".format(module))
    print("=======================================================================")
    print("[{}] Data path: {}".format(module, data_path))
    print("[{}] Output Dir: {}".format(module, output_directory))
    print("[{}] Machine: {}".format(module, machine))
    print("[{}] Iteration: {}".format(module, itr))
    print("[{}] Problem: {}".format(module, problem))
    print("[{}] Regressor: {}".format(module, regressor_name))
    print("[{}] Normalisation: {}".format(module, norm))

    # set data folder
    data_folder = data_path + problem + "/"
    train_file = data_folder + problem + "_TRAIN.ts"
    test_file = data_folder + problem + "_TEST.ts"

    print("[{}] Loading data".format(module))
    X_train, y_train = load_from_tsfile_to_dataframe(train_file)
    X_test, y_test = load_from_tsfile_to_dataframe(test_file)

    print("[{}] X_train: {}".format(module, X_train.shape))
    print("[{}] X_test: {}".format(module, X_test.shape))

    print("[{}] Finding minimum length".format(module))
    min_len = np.inf
    for i in range(len(X_train)):
        x = X_train.iloc[i, :]
        all_len = [len(y) for y in x]
        min_len = min(min(all_len), min_len)
    for i in range(len(X_test)):
        x = X_test.iloc[i, :]
        all_len = [len(y) for y in x]
        min_len = min(min(all_len), min_len)
    print("[{}] Minimum length: {}".format(module, min_len))

    print("[{}] Reshaping data".format(module))
    x_train = process_data(regressor_name, X_train, normalise=norm, min_len=min_len)
    x_test = process_data(regressor_name, X_test, normalise=norm, min_len=min_len)

    print("[{}] X_train: {}".format(module, x_train.shape))
    print("[{}] X_test: {}".format(module, x_test.shape))

    regressor = fit_regressor(output_directory, regressor_name, x_train, y_train, x_test, y_test, itr=itr)

    y_pred = regressor.predict(x_test)
    df_metrics = calculate_regression_metrics(y_test, y_pred)

    print(df_metrics)

    df_metrics.to_csv(output_directory + 'regression_experiment.csv', index=False)
