import multiprocessing
from functools import partial

import numpy as np
from skfda.preprocessing.dim_reduction.projection import FPCA
from sklearn.decomposition import PCA

from smoother.smoothers import bspline
from utils.tools import initialise_multithread


class TimeSeriesTransformer:
    """
    This is a general class for transforming a time series, i.e. extracting features from a time series
    """

    def __init__(self, pca_type="pca", smooth="none",
                 pca_components=10, n_basis=10, bspline_order=4):
        self.train_data = None

        self.pca_type = pca_type
        self.smooth = smooth

        self.n_basis = n_basis
        self.bspline_order = bspline_order
        self.pca_components = pca_components
        self.pca_transformers = []

    def fit(self, series):
        if len(series.shape) == 2:
            series = series.reshape(series.shape[0], series.shape[1], 1)

        self.train_data = series

    def fit_transform(self, series):
        if len(series.shape) == 2:
            series = series.reshape(series.shape[0], series.shape[1], 1)

        self.train_data = series

        n_samples, series_len, n_dim = series.shape
        transformed_shape = (n_samples, self.pca_components * 3 + series_len * 2, n_dim)
        transformed = np.zeros(transformed_shape)

        # create the inputs (N x L series, transformer, dimension)
        inputs = []
        for j in range(n_dim):
            if self.pca_type == "pca":
                transformer = PCA(n_components=self.pca_components)
            elif self.pca_type == "fpca":
                transformer = FPCA(n_components=self.pca_components)
            else:
                transformer = None

            inputs.append((series[:, :, j], transformer, j))

        # start multi-threading
        p = initialise_multithread()
        # series, pca_transformer, dim, pca_type, smooth, n_basis, bspline_order
        outputs = p.starmap(partial(all_fit_transform, pca_type=self.pca_type,
                                    smooth=self.smooth, n_basis=self.n_basis,
                                    bspline_order=self.bspline_order), inputs)
        p.close()

        # expand the outputs
        # transformed, pca_transformer, pca1_transformer, pca2_transformer, dim
        for i in range(len(outputs)):
            dim = outputs[i][4]
            transformed[:, :, dim] = outputs[i][0]
            self.pca_transformers.append(outputs[i][1])

        return transformed

    def transform(self, series):
        pass


class PCATransformer(TimeSeriesTransformer):
    """
    Apply PCA on a set of time series to transform the series into a reduced dimension
    """

    def __init__(self, n_components=10):
        super().__init__()
        self.n_components = n_components
        self.transformer = []

    def fit_transform(self, series):
        """
        Fit a transformer per dimension of the series and transform the series based on the number of coefficients
        :param series: A set of time series with the shape N x L x D
        :return: transformed series with top n_components principal components
        """
        from sklearn.decomposition import PCA
        if len(series.shape) == 2:
            series = series.reshape(series.shape[0], series.shape[1], 1)

        self.train_data = series

        n_samples, series_len, n_dim = series.shape
        transformed = np.zeros((n_samples, self.n_components, n_dim))

        # create the inputs (N x L series, transformer, dimension)
        inputs = []
        for j in range(n_dim):
            transformer = PCA(n_components=self.n_components)
            inputs.append((series[:, :, j], transformer, j))

        # start multi-threading
        p = initialise_multithread()
        num_cores = multiprocessing.cpu_count() - 1
        with multiprocessing.Pool(num_cores) as p:
            outputs = p.starmap(transformer_fit_transform, inputs)
        outputs = p.starmap(transformer_fit_transform, inputs)
        p.close()

        # expand the outputs
        for i in range(len(outputs)):
            dim = outputs[i][2]
            self.transformer.append(outputs[i][1])
            transformed[:, :, dim] = outputs[i][0]

        return transformed

    def transform(self, series):
        """
        Transform the series based on the number of coefficients
        :param series: A set of time series with the shape N x L x D
        :return: transformed series with top n_components principal components
        """
        if len(series.shape) == 2:
            series = series.reshape(series.shape[0], series.shape[1], 1)

        n_samples, series_len, n_dim = series.shape
        transformed = np.zeros((n_samples, self.n_components, n_dim))

        # create the inputs (N x L series, transformer, dimension)
        inputs = []
        for j in range(n_dim):
            transformer = self.transformer[j]
            inputs.append((series[:, :, j], transformer, j))

        # start multi-threading
        p = initialise_multithread()
        outputs = p.starmap(transformer_transform, inputs)
        p.close()

        # expand the outputs
        for i in range(len(outputs)):
            dim = outputs[i][2]
            transformed[:, :, dim] = outputs[i][0]

        return transformed


class FPCATransformer(TimeSeriesTransformer):
    """
    Apply FPCA on a set of time series to transform the series into a reduced dimension
    """

    def __init__(self, n_components=10, n_basis=10, order=4, smooth="none"):
        super().__init__()
        self.n_components = n_components
        self.smooth = smooth
        self.n_basis = n_basis
        self.order = order
        self.transformer = []
        self.sample_points = None
        self.basis = None
        self.basis_fd = None

        if self.smooth == "bspline":
            # n_basis has to be larger or equal to order
            if self.n_basis < self.order:
                self.n_basis = self.order
            # n_components has to be less than n_basis
            self.n_components = min(self.n_basis, self.n_components)

    def fit_transform(self, series):
        """
        Convert the series to its functional form, fit the transformer per dimension and
        transform the series based on the number of coefficients
        :param series: A set of time series with the shape N x L x D
        :return: transformed series with top n_components functional principal components
        """
        from skfda.preprocessing.dim_reduction.projection import FPCA
        from utils.data_processor import to_fd

        if len(series.shape) == 2:
            series = series.reshape(series.shape[0], series.shape[1], 1)

        self.train_data = series

        n_samples, series_len, n_dim = series.shape
        transformed = np.zeros((n_samples, self.n_components, n_dim))

        # create the inputs (N x L series, transformer, dimension)
        inputs = []
        for j in range(n_dim):
            # represent the time series in functional form
            fd = to_fd(series[:, :, j])

            # smooth the series if needed
            if self.smooth == "bspline":
                fd = bspline(fd, self.n_basis, self.order)

            transformer = FPCA(n_components=self.n_components)

            inputs.append((fd, transformer, j))

        # start multi-threading
        p = initialise_multithread()
        outputs = p.starmap(transformer_fit_transform, inputs)
        p.close()

        # expand the outputs
        for i in range(len(outputs)):
            dim = outputs[i][2]
            self.transformer.append(outputs[i][1])
            transformed[:, :, dim] = outputs[i][0]

        return transformed

    def transform(self, series):
        """
        Transform the series based on the number of coefficients
        :param series: A set of time series with the shape N x L x D
        :return: transformed series with top n_components functional principal components
        """
        from utils.data_processor import to_fd

        if len(series.shape) == 2:
            series = series.reshape(series.shape[0], series.shape[1], 1)

        n_samples, series_len, n_dim = series.shape
        transformed = np.zeros((n_samples, self.n_components, n_dim))

        # create the inputs
        inputs = []
        for j in range(n_dim):
            transformer = self.transformer[j]

            # represent the time series in functional data
            fd = to_fd(series[:, :, j])

            # smooth the series if needed
            if self.smooth == "bspline":
                fd = bspline(fd, self.n_basis, self.order)
            inputs.append((fd, transformer, j))

        # start multi-threading
        p = initialise_multithread()
        outputs = p.starmap(transformer_transform, inputs)
        p.close()

        # expands the outputs
        for i in range(len(outputs)):
            dim = outputs[i][2]
            transformed[:, :, dim] = outputs[i][0]

        return transformed


def transformer_fit_transform(series, transformer, dim):
    """
    A function for multi-threading
    :param series: a set of time series to transform with the shape N x L
    :param transformer: transformer to transform the series
    :param dim: dimension of the series
    :return: transformed series
    """
    transformed = transformer.fit_transform(series)
    return transformed, transformer, dim


def transformer_transform(series, transformer, dim):
    """
    A function for multi-threading
    :param series: a set of time series to transform with the shape N x L
    :param transformer: transformer to transform the series
    :param dim: dimension of the series
    :return: transformed series
    """
    transformed = transformer.transform(series)
    return transformed, transformer, dim


def to_fd(series, sample_points=None):
    """
    Convert a set of time series to functional data
    :param series: a set of time series with the shape of N x L
    :param sample_points: sample point of the series
    :return: functional data representation of the time series
    """
    from skfda import FDataGrid

    if sample_points is None:
        sample_points = [x for x in range(series.shape[1])]

    return FDataGrid(series, sample_points)


def all_fit_transform(series, pca_transformer, dim, pca_type, smooth, n_basis, bspline_order):
    """
    A function for multi-threading
    :param series: a set of time series to transform with the shape N x L
    :param pca_transformer: transformer to transform the series
    :param dim: dimension of the series
    :return: transformed series
    """
    # do PCA
    if pca_type == "fpca":
        fd = to_fd(series)
        if smooth == "bspline":
            fd = bspline(fd, n_basis, bspline_order)
        pca_transformed = pca_transformer.fit_transform(fd)
    else:
        pca_transformed = pca_transformer.fit_transform(series)

    # concat all transforms
    transformed = np.concatenate(pca_transformed, axis=1)
    return transformed, pca_transformer, dim
