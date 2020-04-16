import pandas as pd

from utils import tools
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
    if regressor_name == "inception":
        from regressors.inception import InceptionTimeRegressor
        return InceptionTimeRegressor(output_directory, input_shape, verbose)
    if regressor_name == "resnet":
        from regressors.resnet import ResNetRegressor
        return ResNetRegressor(output_directory, input_shape, verbose)
    if regressor_name == "fcn":
        from regressors.fcn import FCNRegressor
        return FCNRegressor(output_directory, input_shape, verbose)
    if regressor_name == "random_forest":
        from regressors.random_forest import RFRegressor
        kwargs = {"n_estimators": 100,
                  "n_jobs": -1,
                  "random_state": 0,
                  "verbose": 2}
        return RFRegressor(output_directory, kwargs=kwargs)
    if regressor_name == "xgboost":
        from regressors.xgboost import XGBoostRegressor
        kwargs = {"n_estimators": 100,
                  "n_jobs": 0,
                  "learning_rate": 0.1,
                  "random_state": 0,
                  "verbosity": 2}
        return XGBoostRegressor(output_directory, kwargs=kwargs)
    if regressor_name == "svr":
        from regressors.svr import SVRRegressor
        kwargs = {"kernel": "rbf",
                  "degree": 3,
                  "gamma": "scale",
                  "coef0": 0.0,
                  "tol": 0.001,
                  "C": 1.0,
                  "epsilon": 0.1,
                  "verbose": verbose}
        return SVRRegressor(output_directory, kwargs=kwargs)
    if regressor_name == "ed1nn":
        from regressors.classic_knn import ClassicKNNRegressor
        kwargs = {"algorithm": "auto",
                  "n_neighbors": 1,
                  "n_jobs": -1,
                  "metric": "euclidean"}
        return ClassicKNNRegressor(output_directory, kwargs=kwargs)
    if regressor_name == "ed5nn":
        from regressors.classic_knn import ClassicKNNRegressor
        kwargs = {"algorithm": "auto",
                  "n_neighbors": 5,
                  "n_jobs": -1,
                  "metric": "euclidean"}
        return ClassicKNNRegressor(output_directory, kwargs=kwargs)


#####################################################
# Program starts
#####################################################
# variables
machine = "pc"
regressors = ["resnet", "fcn", "inception", "xgboost", "random_forest", "svr", "ed1nn", "ed5nn"]
regressor_name = "resnet"
normalise = "none"

# problems = ["LFMC", "AcetoneConcentration", "EthanolConcentration", "BenzeneConcentration",
#             "NewsHeadlineSentiment", "NewsTitleSentiment",
#             "IEEEPPG", "PPGDalia",
#             "CoolerCondition", "ValveCondition", "PumpLeakage", "HydraulicAccumulator"]
problems = ["IEEEPPG", "PPGDalia",
            "CoolerCondition", "ValveCondition", "PumpLeakage", "HydraulicAccumulator",
            "NewsHeadlineSentiment", "NewsTitleSentiment"]
for problem in problems:
    data_folder = tools.get_data_folder(machine) + problem + "/"
    print("[Experiments] Data folder: {}".format(data_folder))

    train_file = data_folder + problem + "_TRAIN.ts"
    test_file = data_folder + problem + "_TEST.ts"

    print("[Experiments] Loading train data")
    X_train, y_train = load_from_tsfile_to_dataframe(train_file)

    print("[Experiments] Loading test data")
    X_test, y_test = load_from_tsfile_to_dataframe(test_file)

    print("[Experiments] Reshaping data")
    X_train = process_data(regressor_name, X_train, normalise=normalise)
    X_test = process_data(regressor_name, X_test, normalise=normalise)

    print("[Experiments] X_train: {}".format(X_train.shape))
    print("[Experiments] X_test: {}".format(X_test.shape))

    input_shape = X_train.shape[1:]
    verbose = True

    itr = "iter_0"
    if normalise == "standard":
        output_directory = "output/standard/"
    elif normalise == "minmax":
        output_directory = "output/minmax/"
    else:
        output_directory = "output/unnorm/"

    output_directory = output_directory + regressor_name + '/' + problem + '/' + itr + '/'
    create_directory(output_directory)

    model = fit_regressor(output_directory, regressor_name, X_train, y_train, X_test, y_test)

    y_pred, test_duration = model.predict(X_test)
    df_metrics = calculate_regression_metrics(y_test, y_pred, test_duration)
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

    print("[Experiments]", df_metrics)
