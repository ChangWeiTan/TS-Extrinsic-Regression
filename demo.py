# Chang Wei Tan, Christoph Bergmeir, Francois Petitjean, Geoff Webb
#
# @article{
#   Tan2020TSER,
#   title={Time Series Extrinsic Regression},
#   author={Tan, Chang Wei and Bergmeir, Christoph and Petitjean, Francois and Webb, Geoffrey I},
#   journal={Data Mining and Knowledge Discovery},
#   pages={1--29},
#   year={2021},
#   publisher={Springer},
#   doi={https://doi.org/10.1007/s10618-021-00745-9}
# }
import argparse

import numpy as np

from utils.data_loader import load_from_tsfile_to_dataframe
from utils.regressor_tools import process_data, fit_regressor, calculate_regression_metrics
from utils.tools import create_directory
from utils.transformer_tools import fit_transformer

module = "RegressionExperiment"

# transformer parameters
transformer_name = "none"  # see transformer_tools.transformers
flatten = False  # if flatten, do not transform per dimension
n_components = 10  # number of principal components
n_basis = 10  # number of basis functions
bspline_order = 4  # bspline order

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", required=False, default="data/")
parser.add_argument("-p", "--problem", required=False, default="Sample")  # see data_loader.regression_datasets
parser.add_argument("-c", "--regressor", required=False, default="rocket")  # see regressor_tools.all_models
parser.add_argument("-i", "--iter", required=False, default=1)
parser.add_argument("-n", "--norm", required=False, default="none")  # none, standard, minmax

arguments = parser.parse_args()

# start the program
if __name__ == '__main__':
    data_path = arguments.data
    problem = arguments.problem  # see data_loader.regression_datasets
    regressor_name = arguments.regressor  # see regressor_tools.all_models
    itr = arguments.iter
    norm = arguments.norm  # none, standard, minmax

    # create output directory
    output_directory = "output/regression/"
    if norm != "none":
        output_directory = "output/regression_{}/".format(norm)
    output_directory = output_directory + regressor_name + '/' + problem + '/itr_' + str(itr) + '/'
    create_directory(output_directory)

    print("=======================================================================")
    print("[{}] Starting Holdout Experiments".format(module))
    print("=======================================================================")
    print("[{}] Data path: {}".format(module, data_path))
    print("[{}] Output Dir: {}".format(module, output_directory))
    print("[{}] Iteration: {}".format(module, itr))
    print("[{}] Problem: {}".format(module, problem))
    print("[{}] Regressor: {}".format(module, regressor_name))
    print("[{}] Transformer: {}".format(module, transformer_name))
    print("[{}] Normalisation: {}".format(module, norm))

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

    # process the data into numpy array with (n_examples, n_timestep, n_dim)
    print("[{}] Reshaping data".format(module))
    x_train = process_data(X_train, normalise=norm, min_len=min_len)
    x_test = process_data(X_test, normalise=norm, min_len=min_len)

    # transform the data if needed
    if transformer_name != "none":
        if transformer_name == "pca":
            kwargs = {"n_components": n_components}
        elif transformer_name == "fpca":
            kwargs = {"n_components": n_components}
        elif transformer_name == "fpca_bspline":
            kwargs = {"n_components": n_components,
                      "n_basis": n_basis,
                      "order": bspline_order,
                      "smooth": "bspline"}
        else:
            kwargs = {}
        x_train, transformer = fit_transformer(transformer_name, x_train, flatten=flatten, **kwargs)
        x_test = transformer.transform(x_test)

    print("[{}] X_train: {}".format(module, x_train.shape))
    print("[{}] X_test: {}".format(module, x_test.shape))

    # fit the regressor
    regressor = fit_regressor(output_directory, regressor_name, x_train, y_train, x_test, y_test, itr=itr)

    # start testing
    y_pred = regressor.predict(x_test)
    df_metrics = calculate_regression_metrics(y_test, y_pred)

    print(df_metrics)

    # save the outputs
    df_metrics.to_csv(output_directory + 'regression_experiment.csv', index=False)
