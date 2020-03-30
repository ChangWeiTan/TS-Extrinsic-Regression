import time

import keras
import numpy as np
from utils.tools import save_logs_for_regression_deep_learning, calculate_regression_metrics, save_test_duration, \
    save_train_duration


class FCNRegressor:
    def __init__(self, output_directory, input_shape, verbose=False, build=True, batch_size=16, nb_epochs=2000):
        self.name = "FCN"
        self.output_directory = output_directory
        if build == True:
            self.batch_size = batch_size
            self.epochs = nb_epochs
            self.params = {"batch_size": batch_size, "nb_epochs": nb_epochs}
            self.verbose = verbose
            self.model = self.build_model(input_shape)
            if (verbose == True):
                self.model.summary()
            self.model.save_weights(self.output_directory + 'model_init.hdf5')
        else:
            self.model = None

        return

    def summary(self):
        print("{}()".format(self.name, self.params))
        if self.model is not None:
            self.model.summary()

    def build_model(self, input_shape):
        if self.verbose > 0:
            print("[{}] Building the model with input shape of {}".format(self.name, input_shape))

        input_layer = keras.layers.Input(input_shape)

        conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation(activation='relu')(conv1)

        conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)

        conv3 = keras.layers.Conv1D(128, kernel_size=3, padding='same')(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)

        gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

        output_layer = keras.layers.Dense(1, activation='linear')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(),
                      metrics=['mae'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                      min_lr=0.0001)

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)

        self.callbacks = [reduce_lr, model_checkpoint]

        return model

    def fit(self, x_train, y_train, x_val, y_val):
        if self.verbose > 0:
            print("[{}] Fitting the regressor with data of {}".format(self.name, x_train.shape))

        batch_size = 16
        nb_epochs = 100

        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

        start_time = time.time()

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)

        y_pred, _ = self.predict(x_train)
        df_metrics = calculate_regression_metrics(y_train, y_pred)

        self.train_duration = time.time() - start_time
        df_metrics["duration"] = self.train_duration
        save_train_duration(self.output_directory + 'train_duration.csv', self.train_duration)
        self.train_metrics = df_metrics

        self.model.save(self.output_directory + 'last_model.hdf5')

        save_logs_for_regression_deep_learning(self.output_directory, hist)

        y_pred, test_duration = self.predict(x_val)

        # save predictions
        np.save(self.output_directory + 'y_pred.npy', y_pred)

        df_metrics = save_logs_for_regression_deep_learning(self.output_directory, hist, y_pred)

        keras.backend.clear_session()

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
        model_path = self.output_directory + 'best_model.hdf5'
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test)

        test_duration = time.time() - start_time
        self.test_duration = test_duration
        save_test_duration(self.output_directory + 'test_duration.csv', test_duration)

        if self.verbose > 0:
            print("[{}] Predicting completed, took {}s".format(self.name, test_duration))

        return y_pred, test_duration
