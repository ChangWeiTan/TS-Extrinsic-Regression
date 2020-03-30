import time

from sklearn.ensemble import RandomForestRegressor

from utils.tools import save_test_duration, calculate_regression_metrics, save_train_duration


class RFRegressor:
    def __init__(self, output_directory, verbose=False, build=True, kwargs=None):
        self.name = "RandomForest"
        self.output_directory = output_directory
        if build:
            if kwargs is None:
                kwargs = {"n_estimators": 100,
                          "n_jobs": -1,
                          "random_state": 0,
                          "verbose": 2}
            self.params = kwargs
            self.build_model(**kwargs)
            self.verbose = verbose

        return

    def summary(self):
        print("{}()".format(self.name, self.params))

    def build_model(self, **kwargs):
        self.model = RandomForestRegressor(**kwargs)
        return self.model

    def fit(self, x_train, y_train, x_val, y_val):
        if self.verbose > 0:
            print("[{}] Fitting the regressor with data of {}".format(self.name, x_train.shape))

        start_time = time.time()

        self.model = self.model.fit(x_train, y_train)

        y_pred, _ = self.predict(x_train)
        df_metrics = calculate_regression_metrics(y_train, y_pred)

        self.train_duration = time.time() - start_time
        df_metrics["duration"] = self.train_duration
        save_train_duration(self.output_directory + 'train_duration.csv', self.train_duration)
        self.train_metrics = df_metrics

        if self.verbose > 0:
            print("[{}] Fitting completed, took {}s".format(self.name, self.train_duration))
            print("[{}] Fitting completed, best RMSE={}, MAE={}".format(self.name,
                                                                        df_metrics["rmse"][0],
                                                                        df_metrics["mae"][0]))
        return df_metrics

    def predict(self, x_test):
        if self.verbose > 0:
            print("[{}] Predicting data of {}".format(self.name, x_test.shape))

        start_time = time.time()

        y_pred = self.model.predict(x_test)

        test_duration = time.time() - start_time
        save_test_duration(self.output_directory + 'test_duration.csv', test_duration)
        self.test_duration = test_duration

        if self.verbose > 0:
            print("[{}] Predicting completed, took {}s".format(self.name, test_duration))

        return y_pred, test_duration
