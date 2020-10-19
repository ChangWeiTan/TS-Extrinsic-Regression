import time

import tensorflow as tf
import numpy as np
import pandas as pd

from models.time_series_models import TimeSeriesRegressor
from utils.tools import save_train_duration, save_test_duration


def plot_epochs_metric(hist, file_name, model, metric='loss'):
    """
    Plot the train/test metrics of Deep Learning models
    :param hist:
    :param file_name:
    :param model:
    :param metric:
    :return:
    """
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(hist.history[metric], label="train")
    if "val_" + metric in hist.history.keys():
        plt.plot(hist.history['val_' + metric], label="val")
    plt.title(model + " " + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


class DLRegressor(TimeSeriesRegressor):
    """
    This is a superclass for Deep Learning models for Regression
    """
    name = "DeepLearningTSR"
    model_init_file = "model_init.h5"
    best_model_file = "best_model.h5"

    def __init__(self, output_directory, input_shape, verbose=False, epochs=200, batch_size=16,
                 loss="mean_squared_error", metrics=None):
        super().__init__(output_directory)
        print('[{}] Creating Regressor'.format(self.name))
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.callbacks = None
        self.hist = None

        self.verbose = verbose
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss = loss
        if metrics is None:
            metrics = ["mae"]
        self.metrics = metrics

        self.model = self.build_model(input_shape)

        self.model.summary()

        self.model.save_weights(self.output_directory + self.model_init_file)

    def build_model(self, input_shape):
        pass

    def fit(self, x_train, y_train, x_val=None, y_val=None, monitor_val=False):
        print('[{}] Training'.format(self.name))

        start_time = time.perf_counter()

        self.X_train = x_train
        self.y_train = y_train
        self.X_val = x_val
        self.y_val = y_val

        epochs = self.epochs
        batch_size = self.batch_size
        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

        file_path = self.output_directory + self.best_model_file
        if (x_val is not None) and monitor_val:
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                          factor=0.5, patience=50,
                                                          min_lr=0.0001)
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=file_path,
                                                               monitor='val_loss',
                                                               save_best_only=True)
        else:
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                          factor=0.5, patience=50,
                                                          min_lr=0.0001)
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=file_path,
                                                               monitor='loss',
                                                               save_best_only=True)
        self.callbacks = [reduce_lr, model_checkpoint]

        # train the model
        if x_val is not None:
            self.hist = self.model.fit(x_train, y_train,
                                       validation_data=(x_val, y_val),
                                       verbose=self.verbose,
                                       epochs=epochs,
                                       batch_size=mini_batch_size,
                                       callbacks=self.callbacks)
        else:
            self.hist = self.model.fit(x_train, y_train,
                                       verbose=self.verbose,
                                       epochs=epochs,
                                       batch_size=mini_batch_size,
                                       callbacks=self.callbacks)

        self.train_duration = time.perf_counter() - start_time

        save_train_duration(self.output_directory + 'train_duration.csv', self.train_duration)

        print('[{}] Training done!, took {}s'.format(self.name, self.train_duration))

    def predict(self, x):
        print('[{}] Predicting'.format(self.name))
        start_time = time.perf_counter()
        model = tf.keras.models.load_model(self.output_directory + self.best_model_file)
        yhat = model.predict(x)

        tf.keras.backend.clear_session()
        test_duration = time.perf_counter() - start_time

        save_test_duration(self.output_directory + 'test_duration.csv', test_duration)

        print('[{}] Prediction done!'.format(self.name))

        return yhat

    def save_logs(self):
        hist_df = pd.DataFrame(self.hist.history)
        hist_df.to_csv(self.output_directory + 'history.csv', index=False)
        if "mae" in hist_df.columns:
            metric = "mae"
        elif "mean_absolute_error" in hist_df.columns:
            metric = "mean_absolute_error"
        else:
            metric = None

        if "val_loss" in hist_df.columns:
            index_best_model = hist_df['val_loss'].idxmin()
            row_best_model = hist_df.loc[index_best_model]

            df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                                         columns=['best_model_train_loss',
                                                  'best_model_val_loss',
                                                  'best_model_train_mae',
                                                  'best_model_val_mae',
                                                  'best_model_learning_rate',
                                                  'best_model_nb_epoch'])

            df_best_model['best_model_train_loss'] = row_best_model['loss']
            if "mae" in hist_df.columns:
                df_best_model['best_model_train_mae'] = row_best_model['mae']
            else:
                df_best_model['best_model_train_mae'] = row_best_model['mean_absolute_error']
            if 'val_loss' in row_best_model.index:
                df_best_model['best_model_val_loss'] = row_best_model['val_loss']
                if "val_mae" in hist_df.columns:
                    df_best_model['best_model_val_mae'] = row_best_model['val_mae']
                else:
                    df_best_model['best_model_val_mae'] = row_best_model['val_mean_absolute_error']

            df_best_model['best_model_nb_epoch'] = index_best_model

            df_best_model.to_csv(self.output_directory + 'df_best_model.csv', index=False)

        # plot losses
        self.metric = plot_epochs_metric(self.hist, self.output_directory + 'epochs_loss.png', model=self.name)
        if metric is not None:
            plot_epochs_metric(self.hist,
                               self.output_directory + 'epochs_' + metric + '.png',
                               model=self.name,
                               metric=metric)
