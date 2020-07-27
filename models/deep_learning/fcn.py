import keras

from models.deep_learning.deep_learning_models import DLRegressor


class FCNRegressor(DLRegressor):
    """
    This is a class implementing the FCN model for time series regression.
    The code is adapted from https://github.com/hfawaz/dl-4-tsc designed for time series classification.
    """

    def __init__(self, output_directory, input_shape, verbose=False, epochs=2000, batch_size=64,
                 loss="mean_squared_error", metrics=None):
        self.name = "FCN"
        super().__init__(output_directory=output_directory,
                         input_shape=input_shape,
                         verbose=verbose,
                         epochs=epochs,
                         batch_size=batch_size,
                         loss=loss,
                         metrics=metrics)

    def build_model(self, input_shape):
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

        model.compile(loss=self.loss,
                      optimizer=keras.optimizers.Adam(),
                      metrics=self.metrics)

        return model
