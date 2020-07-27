import time

from utils.tools import save_train_duration, save_test_duration


class Regressor:
    """
    This is a super class for general machine learning regressors.
    It takes features extracted from the time series as input and passed to a regression model.
    """

    def __init__(self, output_directory):
        self.name = "Regressor"
        self.output_directory = output_directory
        self.train_duration = None
        self.model = None
        self.params = None

    def summary(self):
        print("{}()".format(self.name, self.params))

    def fit(self, x_train, y_train):
        print("[{}] Fitting the regressor with data of {}".format(self.name, x_train.shape))

        start_time = time.time()

        if len(x_train.shape) == 3:
            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])

        self.cv(x_train, y_train)

        self.model.fit(x_train, y_train)

        self.train_duration = time.time() - start_time

        save_train_duration(self.output_directory + 'train_duration.csv', self.train_duration)

        print("[{}] Fitting completed, took {}s".format(self.name, self.train_duration))

    def predict(self, x_test):
        print("[{}] Predicting data of {}".format(self.name, x_test.shape))

        start_time = time.time()

        y_pred = self.model.predict(x_test)

        test_duration = time.time() - start_time

        save_test_duration(self.output_directory + 'test_duration.csv', test_duration)

        print("[{}] Predicting completed, took {}s".format(self.name, test_duration))

        return y_pred

    def cv(self, x_train, y_train):
        pass


class SVRRegressor(Regressor):
    """
    This is a wrapper for SupportVectorRegression (SVR) model.
    """

    def __init__(self, output_directory, verbose=0, kwargs=None):
        super().__init__(output_directory)
        self.name = "SVR"

        if kwargs is None:
            kwargs = {"kernel": "rbf",
                      "degree": 3,
                      "gamma": "scale",
                      "coef0": 0.0,
                      "tol": 0.001,
                      "C": 1.0,
                      "epsilon": 0.1,
                      "verbose": verbose}
        self.params = kwargs
        self.build_model(**kwargs)

        return

    def build_model(self, **kwargs):
        from sklearn.svm import SVR

        self.model = SVR(**kwargs)
        return self.model


class RFRegressor(Regressor):
    """
    This is a wrapper for RandomForest model.
    """

    def __init__(self, output_directory, verbose=0, kwargs=None):
        super().__init__(output_directory)
        self.name = "RandomForest"

        if kwargs is None:
            kwargs = {"n_estimators": 100,
                      "n_jobs": -1,
                      "random_state": 0,
                      "verbose": verbose,
                      # Parameters that we are going to tune.
                      "max_depth": 6,
                      "min_child_weight": 1,
                      "eta": .3,
                      "subsample": 1,
                      "colsample_bytree": 1,
                      # Other parameters
                      "objective": "reg:linear", }

        self.params = kwargs
        self.build_model(**kwargs)

        return

    def build_model(self, **kwargs):
        from sklearn.ensemble import RandomForestRegressor

        self.model = RandomForestRegressor(**kwargs)
        return self.model


class XGBoostRegressor(Regressor):
    """
    This is a wrapper for XGBoost.
    """

    def __init__(self, output_directory, verbose=0, kwargs=None):
        super().__init__(output_directory)
        self.name = "XGBoost"

        if kwargs is None:
            kwargs = {"n_estimators": 100,
                      "n_jobs": 0,
                      "learning_rate": 0.1,
                      "random_state": 0,
                      "verbosity  ": verbose}
        self.params = kwargs
        self.gridsearch_params = [
            (max_depth, min_child_weight, subsample, colsample, eta)
            for max_depth in range(9, 12)
            for min_child_weight in range(5, 8)
            for subsample in [i / 10. for i in range(7, 11)]
            for colsample in [i / 10. for i in range(7, 11)]
            for eta in [.3, .2, .1, .05, .01, .005]
        ]
        self.build_model(**kwargs)

        return

    def build_model(self, **kwargs):
        from xgboost import XGBRegressor

        self.model = XGBRegressor(**kwargs)

        return self.model
