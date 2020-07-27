# Chang Wei Tan, Christoph Bergmeir, Francois Petitjean, Geoff Webb
#
# @article{Tan2020Time,
#   title={Time Series Regression},
#   author={Tan, Chang Wei and Bergmeir, Christoph and Petitjean, Francois and Webb, Geoffrey I},
#   journal={arXiv preprint arXiv:2006.12672},
#   year={2020}
# }

import numpy as np

from utils.data_loader import load_from_tsfile_to_dataframe, regression_datasets
from utils.regressor_tools import process_data, fit_regressor, calculate_regression_metrics, all_models
from utils.tools import create_directory

module = "RegressionExperiment"
data_path = "data/"
problems = ["Sample"]       # see data_loader.regression_datasets
regressors = all_models     # see regressor_tools.all_models
iterations = [1]
norm = "none"               # none, standard, minmax

output_path = "output/regression/"
if __name__ == '__main__':
    # for each problem
    for problem in problems:
        print("#########################################################################")
        print("[{}] Starting Experiments".format(module))
        print("#########################################################################")
        print("[{}] Data path: {}".format(module, data_path))
        print("[{}] Problem: {}".format(module, problem))

        # set data folder, train & test
        data_folder = data_path + problem + "/"
        train_file = data_folder + problem + "_TRAIN.ts"
        test_file = data_folder + problem + "_TEST.ts"

        # loading the data. X_train and X_test are dataframe of N x n_dim
        print("[{}] Loading data".format(module))
        X_train, y_train = load_from_tsfile_to_dataframe(train_file)
        X_test, y_test = load_from_tsfile_to_dataframe(test_file)

        print("[{}] X_train: {}".format(module, X_train.shape))
        print("[{}] X_test: {}".format(module, X_test.shape))

        # in case there are different lengths in the dataset, we need to consider that.
        # assume that all the dimensions are the same length
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

        # process the data into numpy array
        print("[{}] Reshaping data".format(module))
        x_train = process_data(X_train, normalise=norm, min_len=min_len)
        x_test = process_data(X_test, normalise=norm, min_len=min_len)

        print("[{}] X_train: {}".format(module, x_train.shape))
        print("[{}] X_test: {}".format(module, x_test.shape))

        for regressor_name in regressors:
            print("[{}] Regressor: {}".format(module, regressor_name))
            for itr in iterations:
                # create output directory
                output_directory = "output/regression/"
                if norm != "none":
                    output_directory = "output/regression_{}/".format(norm)
                output_directory = output_directory + regressor_name + '/' + problem + '/itr_' + str(itr) + '/'
                create_directory(output_directory)

                print("[{}] Iteration: {}".format(module, itr))
                print("[{}] Output Dir: {}".format(module, output_directory))

                # fit the regressor
                regressor = fit_regressor(output_directory, regressor_name, x_train, y_train, x_test, y_test, itr=itr)

                # start testing
                y_pred = regressor.predict(x_test)
                df_metrics = calculate_regression_metrics(y_test, y_pred)

                print(df_metrics)

                # save the outputs
                df_metrics.to_csv(output_directory + 'regression_experiment.csv', index=False)
