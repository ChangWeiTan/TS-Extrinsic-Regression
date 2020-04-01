import time

import keras
import numpy as np

from utils.tools import save_logs_for_regression_deep_learning, calculate_regression_metrics, save_test_duration, \
    save_train_duration


class ResNetRegressor:
    def __init__(self, output_directory, input_shape, verbose=False, build=True, batch_size=64, nb_epochs=1500):
        self.name = "ResNet"
        self.output_directory = output_directory
        self.verbose = verbose
        if build == True:
            self.batch_size = batch_size
            self.epochs = nb_epochs
            self.params = {"batch_size": batch_size, "nb_epochs": nb_epochs}
            self.model = self.build_model(input_shape)
            if (self.verbose == True):
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

        n_feature_maps = 64

        input_layer = keras.layers.Input(input_shape)

        # BLOCK 1

        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)

        # BLOCK 2

        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = keras.layers.Activation('relu')(output_block_2)

        # BLOCK 3

        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = keras.layers.BatchNormalization()(output_block_2)

        output_block_3 = keras.layers.add([shortcut_y, conv_z])
        output_block_3 = keras.layers.Activation('relu')(output_block_3)

        # FINAL

        gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

        output_layer = keras.layers.Dense(1, activation='linear')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(),
                      metrics=['mae'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)

        self.callbacks = [reduce_lr, model_checkpoint]

        return model

    def fit(self, x_train, y_train, x_val, y_val):
        if self.verbose > 0:
            print("[{}] Fitting the regressor with data of {}".format(self.name, x_train.shape))

        # x_val and y_val are only used to monitor the test loss and NOT for training

        batch_size = 64
        nb_epochs = 1500

        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

        start_time = time.time()

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)

        self.train_duration = time.time() - start_time

        y_pred, _ = self.predict(x_train)
        df_metrics = calculate_regression_metrics(y_train, y_pred)
        df_metrics["duration"] = self.train_duration
        save_train_duration(self.output_directory + 'train_duration.csv', self.train_duration)
        self.train_metrics = df_metrics

        self.model.save(self.output_directory + 'last_model.hdf5')

        save_logs_for_regression_deep_learning(self.output_directory, hist)

        keras.backend.clear_session()

        if self.verbose > 0:
            print("[{}] Fitting completed, took {}s".format(self.name, self.train_duration))
            print("[{}] Fitting completed, RMSE={}, MAE={}".format(self.name,
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
        save_test_duration(self.output_directory + 'test_duration.csv', test_duration)

        if self.verbose > 0:
            print("[{}] Predicting completed, took {}s".format(self.name, test_duration))

        return y_pred, test_duration
