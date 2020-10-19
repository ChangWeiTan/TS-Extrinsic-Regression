# Chang Wei Tan, Christoph Bergmeir, Francois Petitjean, Geoff Webb
#
# @article{Tan2020Time,
#   title={Time Series Regression},
#   author={Tan, Chang Wei and Bergmeir, Christoph and Petitjean, Francois and Webb, Geoffrey I},
#   journal={arXiv preprint arXiv:2006.12672},
#   year={2020}
# }
import getopt
import sys

import numpy as np

from utils.data_loader import load_from_tsfile_to_dataframe
from utils.regressor_tools import process_data, fit_regressor, calculate_regression_metrics
from utils.tools import create_directory
from utils.transformer_tools import fit_transformer

module = "RegressionExperiment"
data_path = "data/"
problem = "Sample"  # see data_loader.regression_datasets
regressor_name = "rocket"  # see regressor_tools.all_models
transformer_name = "none"  # see transformer_tools.transformers
itr = 1
norm = "none"  # none, standard, minmax

# transformer parameters
flatten = False  # if flatten, do not transform per dimension
n_components = 10  # number of principal components
n_basis = 10  # number of basis functions
bspline_order = 4  # bspline order

# parse arguments
try:
    opts, args = getopt.getopt(sys.argv[1:], "hd:p:r:i:n:m:",
                               ["data_path=", "problem=", "regressor=", "iter=", "norm="])
except getopt.GetoptError:
    print("demo.py -d <data_path> -p <problem> -r <regressor> -i <iteration> -n <normalisation>")
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print("demo.py -d <data_path> -p <problem> -s <regressor> -i <iteration> -n <normalisation>")
        sys.exit()
    elif opt in ("-d", "--data"):
        data_path = arg
    elif opt in ("-p", "--problem"):
        problem = arg
    elif opt in ("-r", "--regressor"):
        regressor_name = arg
    elif opt in ("-i", "--iter"):
        itr = arg
    elif opt in ("-n", "--norm"):
        norm = arg

# start the program
if __name__ == '__main__':
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

    # process the data into numpy array
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
