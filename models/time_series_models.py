class TimeSeriesRegressor:
    """
    This is a super class for time series regressors
    """

    def __init__(self, output_directory):
        self.output_directory = output_directory
        self.train_duration = None
        self.name = "TimeSeriesRegressor"
        pass

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        pass

    def predict(self, x):
        pass
