import math
import os
import pickle
import warnings
from typing import Callable, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from pandas.core.frame import DataFrame

# Suppress RankWarning
warnings.simplefilter("ignore", np.RankWarning)

# -------------------------------------------------------------------------

"""
Data Science from Scratch, Second Edition, by Joel Grus (O'Reilly).Copyright 2019 Joel Grus, 978-1-492-04113-9
"""


def minibatches(dataset: List, batch_size: int, shuffle: bool = True) -> List:
    """Generates 'batch_size'-sized minibatches from the dataset

    Parameters
    ----------
    dataset : `List`
        The data to be divided into mini-batch.
    batch_size : `int`
        Specifies the size of each mini-batch.
    shuffle : `bool`
        If set `True`, the data will be shuffled before dividing it into mini-batches.

    Returns
    -------
    `List[List]`
        A list of lists containing the mini-batches. Each sublist is a separate mini-batch with length `batch_size`.
    """

    # start indexes 0, batch_size, 2 * batch_size, ...
    batch_starts = [start for start in range(0, len(dataset), batch_size)]

    if shuffle:
        np.random.shuffle(batch_starts)  # shuffle the batches

    for start in batch_starts:
        end = start + batch_size
        yield dataset[start:end]


def difference_quotient(f: Callable, x: float, h: float) -> Callable:
    """Calculates the difference quotient of `f` evaluated at `x` and `x + h`

    Parameters
    ----------
    `f(x)` : `Callable`
        function.
    x : `float`
        Independent term.
    h : `float`
        Step size.

    Returns
    -------
    `(f(x + h) - f(x)) / h` : `float`
        Difference quotient of `f` evaluated at `x`.

    """

    return (f(x + h) - f(x)) / h


def partial_difference_quotient(f: Callable, v: np.ndarray, i: int, h: float) -> np.ndarray:
    """Calculates the partial difference quotient of `f`

    Parameters
    ----------
    `f(x0,...,xi-th)` : `Callable` function
        Function to differentiate.
    v : `Vector` | `np.array`
        1D array representing vector `v=(x0,...,xi)`.
    h : `float`
        Step size.

    Returns
    -------
    `(f(w) - f(v)) / h` : `np.array`
        the `i-th` partial difference quotient of `f` at `v`

    """

    w = [
        v_j + (h if j == i else 0) for j, v_j in enumerate(v)  # add h to just the ith element of v
    ]
    return (f(w) - f(v)) / h


def estimate_gradient(f: Callable, v: np.ndarray, h: float = 1e-4) -> List[np.ndarray]:
    """Calculates the gradient of `f` at `v`

    Parameters
    ----------
    `f(x0,...,xi-th)` : `Callable` function
        Function to differentiate.
    v : `Vector` | `np.array`
        1D array representing vector `v=(x0,...,xi)`.
    h : `float`. By default it is set to `1e-4`
        The step size used to approximate the derivative.

    Returns
    -------
    grad_f : `List[np.array]`
        A list containing the estimated gradients of each component of `f` evaluated at `v`.
    """
    return [partial_difference_quotient(f, v, i, h) for i in range(len(v))]


# -------------------------------------------------------------------------


def generate_feature_yaml(
    df: DataFrame, ignore_features: List[str] = None, yaml_string: bool = False
) -> Dict | str:
    """
    Generate a YAML string containing information about ordinal, numeric, and categorical features
    based on the given DataFrame.

    Parameters
    ----------
    df : `pd.DataFrame`
        The DataFrame containing the data.
    ignore_features : `List[`str`]`
        A list of features to ignore.
    yaml_string : `bool`
        If `True`, return the result as a YAML formatted string. Otherwise, return it as a dictionary. Default is `False`.

    Returns
    -------
    feature_info : `Dict` | `str`
        A dictionary with four keys ('ordinal_features', 'numeric_features', 'categorical_features', 'ignore_features')
        mapping to lists of feature names. Or a YAML formatted string if `yaml_string` is `True`.
    """
    ignore_features = ignore_features or []
    feature_info = {
        "ordinal_features": [],
        "numeric_features": [],
        "categorical_features": [],
        "ignore_features": ignore_features,
    }

    for col in df.columns:
        if col in ignore_features:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            if pd.api.types.is_integer_dtype(df[col]) or pd.api.types.is_float_dtype(df[col]):
                feature_info["numeric_features"].append(col)
            elif pd.api.types.is_bool_dtype(df[col]):
                feature_info["ordinal_features"].append(col)  # Assuming bool can be ordinal
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            feature_info["categorical_features"].append(col)
        else:
            print(f"Unknown type for feature {col}")

    if yaml_string:
        return yaml.dump(feature_info, default_flow_style=False)

    return feature_info


# a function that calculates the percentage of missing values per column is defined
def cal_missing_values(df: DataFrame) -> None:
    """Calculate the percentage of missing (`NaN`/`NaT`) values per column in a dataframe.

    Parameters
    ----------
    df : `DataFrame`
        The input dataframe.

    Returns
    -------
    `None`
        Prints out a table with columns as index and percentages of missing values as data.
    """

    col = df.columns
    print("Total size : ", "{:,}".format(len(df)))
    for i in col:
        print(
            str(i) + " : " f"{(df.isnull().sum()[i]/(df.isnull().sum()[i]+df[i].count()))*100:.2f}%"
        )


def cdf(
    x: np.ndarray, poly: int = 9, inv: bool = False, plot: bool = False, savename: str = None
) -> tuple:
    """Calculates the cumulative distribution function of the data.

    Parameters
    ----------
    x : `np.array`
        An array containing the data.
    poly : `int`
        Degree of the polynomial fit. By default it is set to `9`.
    inv : `bool`
        If True, calculate the inverse CDF (quantile function).
    plot : `bool`
        If True, plot the results.
    savename : `str`, optional
        Filename to save the plot.

    Returns
    -------
    fit : `np.poly1d`
        Polynomial fit of the CDF or quantile function.
    cdf_values : `np.array`
        Cumulative distribution values.
    sorted_x : `np.array`
        Sorted input data.
    """

    if len(x) == 0:
        raise ValueError("Input array 'x' must not be empty.")

    cdf_values = np.cumsum(x) / np.sum(x)
    sorted_x = np.sort(x)

    # Calculate the CDF or inverse CDF (quantile function)
    probabilities = np.linspace(0, 1, len(sorted_x))

    if inv:
        fit = np.polyfit(probabilities, sorted_x, poly)
        f = np.poly1d(fit)
        plot_label = "Quantile Function"
        x_values = probabilities
        y_values = sorted_x
    else:
        fit = np.polyfit(sorted_x, probabilities, poly)
        f = np.poly1d(fit)
        plot_label = "Cumulative Distribution Function"
        x_values = sorted_x
        y_values = cdf_values

    if plot:
        plt.figure()
        plt.plot(x_values, y_values, "o", label="data")
        plt.plot(x_values, f(x_values), "r--", label="fit")
        plt.title(plot_label)
        plt.xlabel("Probability" if inv else "Value")
        plt.ylabel("Value" if inv else "Probability")
        plt.legend()
        if savename:
            plt.savefig(savename, dpi=300)
        plt.show()

    return f, cdf_values, sorted_x


def calculate_probability(x: np.ndarray, points: int = 1, cond: bool = True) -> np.ndarray:
    """Calculates the probability of the data based on the CDF fit.

    Parameters
    ----------
    x : `np.array`
        An array containing the data.
    points : `int`
        Number of points to consider for the final probability calculation.
    cond : `bool`
        Condition to use product (True) or sum (False) for the final probability check.

    Returns
    -------
    p : `np.array`
        Array containing the probabilities of the data.
    """

    if len(x) == 0:
        raise ValueError("Input array 'x' must not be empty.")

    fit, _, sorted_x = cdf(x)
    p = fit(x)

    # Validate probability values
    if cond:
        prob_value = np.prod(p[-points])
        message = "product"
    else:
        prob_value = np.sum(p[-points])
        message = "sum"

    if 0 <= prob_value <= 1:
        print(f"The model has a probability of {prob_value * 100:.2f}% based on the {message}.")
    else:
        print("\nThe probability of the data cannot be calculated.\n")

    return p


class CorrelationBase:
    """Base class for correlation calculations."""

    __slots__ = ["x", "y", "result", "z"]

    def __init__(self, x: np.ndarray, y: Union[np.ndarray, None] = None):
        self.x = x
        self.y = y if y is not None else x  # Default to autocorrelation if y is not provided
        self._compute_correlation()
        self.z = self.result[self.result.size // 2 :]
        self.z /= np.abs(self.z).max()

    def _compute_correlation(self):
        """Compute the correlation between x and y (or x with itself for autocorrelation)."""
        self.result = np.correlate(self.x, self.y, mode="full")

    def plot(self):
        """Plot the correlation or autocorrelation."""
        plt.plot(range(len(self.z)), self.z, label=self._get_label())
        plt.legend()
        plt.show()

    def _get_label(self) -> str:
        return "Autocorrelation" if np.array_equal(self.x, self.y) else "Correlation"

    def __call__(self):
        """Return the computed correlation or autocorrelation."""
        return self.z


class Correlation(CorrelationBase):
    """Calculates the cross-correlation of two datasets.

    Parameters
    ----------
    x : `np.ndarray`
        An array containing the first dataset.
    y : `np.ndarray`
        An array containing the second dataset.

    Returns
    -------
    z : `np.ndarray`
        An array containing the correlation of `x` and `y`.

    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        super().__init__(x, y)


class AutoCorrelation(CorrelationBase):
    """Calculates the autocorrelation of a dataset.

    Parameters
    ----------
    x : `np.ndarray`
        An array containing the data.

    Returns
    -------
    z : `np.ndarray`
        An array containing the autocorrelation of the data.
    """

    def __init__(self, x: np.ndarray):
        super().__init__(x)


def fft_denoise(
    dataset: np.ndarray, sigma: float = 0, mode: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Performs noise removal using the Fast Fourier Transform.

    Parameters
    ----------
    dataset : `np.ndarray`
        An array containing the noised data. Expected shape (num_samples, num_points).
    sigma : `float`, default=0
        A float between 0 and 1 representing the threshold for noise filtering.
    mode : `bool`, default=True
        If True, print progress messages.

    Returns
    -------
    denoised_dataset : `np.ndarray`
        An array containing the denoised data with the same shape as `dataset`.
    periods : `np.ndarray`
        Array of estimated periods for each sample in `dataset`.
    """

    if not (0 <= sigma <= 1):
        raise ValueError("sigma must be between 0 and 1")

    num_samples, n_points = dataset.shape
    denoised_dataset = np.zeros_like(dataset)
    periods = np.zeros(num_samples)

    # Precompute values that do not change within the loop
    freq = (1 / n_points) * np.arange(n_points)
    L = np.arange(1, np.floor(n_points / 2), dtype=int)

    for i in range(num_samples):
        fhat = np.fft.fft(dataset[i, :], n_points)
        PSD = fhat * np.conj(fhat) / n_points
        threshold = np.mean(PSD) + sigma * np.std(PSD)
        indices = PSD > threshold

        # Zero out all others in frequency domain
        PSDclean = PSD * indices
        fhat_cleaned = fhat * indices

        # Inverse FFT for filtered time signal
        denoised_signal = np.fft.ifft(fhat_cleaned).real
        denoised_dataset[i, :] = denoised_signal

        # Calculate the period of the signal
        peak_index = L[np.argmax(np.abs(fhat[L]))]
        periods[i] = 1 / (2 * freq[peak_index])

        if mode:
            print(f"The {i+1}-th row of the dataset has been denoised.")
            print(f"The estimated period is {round(periods[i], 4)}")

    return denoised_dataset, periods


def get_period(dataset: np.ndarray) -> float:
    """Calculates the periodicity of a `dataset`.

    Parameters
    ----------
    dataset : `ndarray`
        the `dataset` describing the function over which the period is calculated

    Returns
    -------
    period : `float`
        period of the function described by the `dataset`
    """
    n = dataset.size

    # Ensure there are enough points for FFT analysis
    if n < 2:
        raise ValueError("Dataset must contain at least two points.")

    # Compute the FFT and PSD
    fhat = np.fft.rfft(dataset)  # Use rfft for real-valued input to save computation
    freqs = np.fft.rfftfreq(n)  # Get only positive frequencies

    # Calculate the Power Spectral Density (PSD)
    PSD = np.abs(fhat) ** 2 / n

    # Remove the first frequency component (DC component)
    PSD[0] = 0

    # Find the index of the maximum PSD value, excluding the DC component
    max_psd_index = np.argmax(PSD)

    # Calculate the period based on the corresponding frequency
    dominant_freq = freqs[max_psd_index]
    if dominant_freq == 0:
        raise ValueError("No significant periodic component found in the dataset.")

    period = 1 / dominant_freq

    return period


def sigmoide_inv(y: float) -> float:
    """Calculates the inverse of the sigmoid function

    Parameters
    ----------
    y : `float`
        the number to evaluate the function

    Returns
    -------
    `float`
        value of evaluated function
    """

    return math.log(y / (1 - y))


def sigmoide(x: float) -> float:
    """The sigmoid function"""
    return 1 / (1 + math.exp(-x))


class LogisticRegression:
    """class implementing multiple logistic regression"""

    __slots__ = ["importance", "X", "y", "w"]

    def __init__(self) -> None:
        """The class initializer"""

        self.importance = []

    def fit(self, dataset: np.ndarray, values: np.ndarray) -> None:
        """Performs linear multiple model training

        Parameters
        ----------
        dataset : `np.array`
            An array containing the scaled data.
        values : `np.ndarray`
            A set of values returned by the linear function.

        Returns
        -------
        importance : `np.array`
            An array containing the importance of each feature.

        """

        self.X = dataset
        self.y = values

        U, S, VT = np.linalg.svd(self.X, full_matrices=False)

        inverse_sig = np.vectorize(sigmoide_inv)
        self.w = (VT.T @ np.linalg.inv(np.diag(S)) @ U.T).T @ inverse_sig(self.y)

        if self.y.shape[1] > 1:
            for row in self.w:
                self.importance.append(np.around(np.max(row), decimals=8))
        else:
            for i in range(self.X.shape[0]):
                a = np.around(self.w[i], decimals=8)
                self.importance.append(a)

    def predict(self, datapoints: np.ndarray) -> np.ndarray:
        """
        Performs predictions for a set of points

        Parameters
        ----------
        datapoints : `np.array`
            An array containing the values of the independent variable.

        """
        sig = np.vectorize(sigmoide)

        return sig(np.array(self.importance) @ datapoints)

    def get_importances(self, print_important_features: bool = False) -> np.ndarray:
        """
        Returns the important features

        Parameters
        ----------
        print_important_features : `bool`
            determines whether or not are printed on the screen. By default it is set to `False`.

        Returns
        -------
        importance : `np.array`
            An array containing the importance of each feature.


        """
        if print_important_features:
            for i, a in enumerate(self.importance):
                print(f"The importance of the {i+1} feature is {a}")
        return np.array(self.importance)


class LinearRegression:
    """class implementing multiple linear regression"""

    __slots__ = ["importance", "X", "y", "w"]

    def __init__(self) -> None:
        """The class initializer"""

        self.importance = []

    def fit(self, dataset: np.ndarray, values: np.ndarray, verbose: bool = False) -> None:
        """Performs linear multiple model training

        Parameters
        ----------
        dataset : `np.array`
            An array containing the scaled data.
        values : `np.ndarray`
            A set of values returned by the linear function.

        Returns
        -------
        importance : `np.array`
            An array containing the importance of each feature.

        """

        self.X = dataset
        self.y = values

        U, S, VT = np.linalg.svd(self.X, full_matrices=False)
        self.w = (VT.T @ np.linalg.inv(np.diag(S)) @ U.T).T @ self.y

        for i in range(self.X.shape[0]):
            a = np.around(self.w[i], decimals=8)
            self.importance.append(a)

        if verbose:
            print("\nSummary:")
            print("--------")
            print("\nParameters:", np.array(self.importance).shape)
            print("RMSE: {:.4f}".format(mean_square_error(self.y, self.predict(self.X))))

    def predict(self, datapoints: np.ndarray) -> np.ndarray:
        """
        Performs predictions for a set of points

        Parameters
        ----------
        datapoints : `np.array`
            An array containing the values of the independent variable.

        """
        return np.array(self.importance) @ datapoints

    def get_importances(self, print_important_features: bool = False) -> np.ndarray:
        """
        Returns the important features

        Parameters
        ----------
        print_important_features : `bool`
            determines whether or not are printed on the screen. By default it is set to `False`.

        Returns
        -------
        importance : `np.array`
            An array containing the importance of each feature.


        """
        if print_important_features:
            for i, a in enumerate(self.importance):
                print(f"The importance of the {i+1} feature is {a}")
        return np.array(self.importance)


def cal_average(y: np.ndarray, alpha: float = 1):
    """Calculates the moving average of the data

    Parameters
    ----------
    y : `np.array`
        An array containing the data.
    alpha : `float`
        A `float` between `0` and `1`. By default it is set to `1`.

    Returns
    -------
    average : `float`
        The average of the data.

    """

    n = int(alpha * len(y))
    w = np.ones(n) / n
    average = np.convolve(y, w, mode="same") / np.convolve(np.ones_like(y), w, mode="same")
    return average


class DataScaler:
    """numpy array `scaler` and `rescaler`"""

    __slots__ = ["dataset_", "_n", "data_scaled", "values", "transpose", "inv_fitting"]

    def __init__(self, dataset: np.ndarray, n: int = 1) -> None:
        """Initializes the parameters required for scaling the data"""
        self.dataset_ = dataset.copy()
        self._n = n

    def rescale(self, dataset_: np.ndarray | None = None) -> np.ndarray:
        """Perform a standard rescaling of the data

        Returns
        -------
        data_scaled : `np.array`
            An array containing the scaled data.
        """
        if isinstance(dataset_, np.ndarray):
            data_scaled = np.copy(dataset_)
            mu = self.values[0]
            sigma = self.values[1]
            f = self.values[2]
            data_scaled = data_scaled.reshape((self.dataset_.shape[0], -1))
            for i in range(self.dataset_.shape[0]):
                if self._n != None:
                    poly = f[i](self.inv_fitting[i](data_scaled[i]))
                    data_scaled[i] += -poly
                data_scaled[i] = 2 * ((data_scaled[i] - mu[i]) / sigma[i]) - 1
            return data_scaled
        else:
            self.data_scaled = np.copy(self.dataset_.copy())

        mu = []
        sigma = []
        fitting = []
        self.inv_fitting = []

        try:
            xaxis = range(self.dataset_.shape[1])
        except:
            error_type = "IndexError"
            msg = "Trying to access an item at an invalid index."
            print(f"{error_type}: {msg}")
            return None
        if self.dataset_.shape[0] > self.dataset_.shape[1]:
            self.dataset_ = self.dataset_.T
            self.transpose = True
        else:
            self.transpose = False
        for i in range(self.dataset_.shape[0]):
            if self._n != None:
                fit = np.polyfit(xaxis, self.dataset_[i, :], self._n)
                inv_fit = np.polyfit(self.dataset_[i, :], xaxis, self._n)
                f = np.poly1d(fit)
                poly = f(xaxis)
                fitting.append(f)
                self.inv_fitting.append(inv_fit)
                self.data_scaled[i, :] += -poly
            else:
                fitting.append(0.0)
                self.inv_fitting.append(0.0)
            mu.append(np.min(self.data_scaled[i, :]))
            if np.max(self.data_scaled[i, :]) != 0:
                sigma.append(np.max(self.data_scaled[i, :]) - mu[i])
            else:
                sigma.append(1)

            self.data_scaled[i, :] = 2 * ((self.data_scaled[i, :] - mu[i]) / sigma[i]) - 1

        self.values = [mu, sigma, fitting]

        return self.data_scaled

    def scale(self, dataset_: np.ndarray) -> np.ndarray:
        """Performs the inverse operation to the rescale function

        Parameters
        ----------
        dataset_ : `np.array`
            An array containing the scaled values.

        Returns
        -------
        dataset_ : `np.array`
            An array containing the rescaled data.
        """
        if self.transpose:
            dataset_ = dataset_.T
        for i in range(dataset_.shape[0]):
            dataset_[i, :] += 1
            dataset_[i, :] /= 2
            dataset_[i, :] = dataset_[i, :] * self.values[1][i]
            dataset_[i, :] += self.values[0][i]
            dataset_[i, :] += self.values[2][i](range(dataset_.shape[1]))

        return dataset_


def generate_series(n: int, n_steps: int, incline: bool = True):
    """Function that generates `n` series of length `n_steps`"""
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, n, 1)

    if incline:
        slope = np.random.rand(n, 1)
    else:
        slope = 0.0
        offsets2 = 1

    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  # wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20))  # + wave 2
    series += 0.7 * (np.random.rand(n, n_steps) - 0.5)  # + noise
    series += 5 * slope * time + 2 * (offsets2 - offsets1) * time ** (1 - offsets2)
    series = series
    return series.astype(np.float32)


def mean_square_error(y_true: np.ndarray, y_pred: np.ndarray, print_error: bool = False):
    """Calculates the Root Mean Squared Error

    Parameters
    ----------
    y_true : `np.array`
        An array containing the true values.
    y_pred : `np.array`
        An array containing the predicted values.

    Returns
    -------
    RMSE : `float`
        The Root Mean Squared Error.

    """
    if print_error:
        print(f"The RMSE is {np.sqrt(np.mean((y_true - y_pred)**2))}")

    return np.sqrt(np.mean((y_true - y_pred) ** 2))


class DataFrameEncoder:
    """Allows encoding and decoding Dataframes"""

    __slots__ = [
        "_df",
        "_names",
        "_encode_columns",
        "encoding_list",
        "decoding_list",
        "median_list",
    ]

    def __init__(self, data: DataFrame) -> None:
        """Sets the columns of the `DataFrame`"""
        self._df = data.copy()
        self._names = data.columns
        self._encode_columns = []
        self.encoding_list = []
        self.decoding_list = []
        self.median_list = []

    def load_config(self, path_to_dictionaries: str = "./", **kwargs) -> None:
        """Loads dictionaries from a given directory

        Keyword Arguments:
        ----------
        - dictionary_name (`str`): An optional string parameter. By default it is set to `labelencoder_dictionary`
        """
        dictionary_name = (
            kwargs["dictionary_name"] if "dictionary_name" in kwargs else "labelencoder_dictionary"
        )
        with open(os.path.join(path_to_dictionaries, dictionary_name + ".pkl"), "rb") as file:
            labelencoder = pickle.load(file)
        self.encoding_list = labelencoder[0]
        self.decoding_list = labelencoder[1]
        self._encode_columns = labelencoder[2]
        self.median_list = labelencoder[3]
        print("Configuration successfully uploaded")

    def train(self, path_to_save: str, **kwargs) -> None:
        """Trains the encoders and decoders using the `DataFrame`"""
        save_mode = kwargs["save_mode"] if "save_mode" in kwargs else True
        dictionary_name = (
            kwargs["dictionary_name"] if "dictionary_name" in kwargs else "labelencoder_dictionary"
        )
        norm_method = kwargs["norm_method"] if "norm_method" in kwargs else "None"
        for i in self._names:
            if self._df[i].dtype == "object":
                self._encode_columns.append(i)
                column_index = range(len(self._df[i].unique()))
                column_keys = self._df[i].unique()
                encode_dict = dict(zip(column_keys, column_index))
                decode_dict = dict(zip(column_index, column_keys))
                self._df[i] = self._df[i].apply(
                    self._code_transformation_to, dictionary_list=encode_dict
                )
                if len(self._df[i].unique()) > 1:
                    median_value = len(self._df[i].unique()) // 2
                else:
                    median_value = 1.0
                if norm_method == "median":
                    self._df[i] = self._df[i].astype("float64")
                    self._df[i] = self._df[i] / median_value
                    self.median_list.append(median_value)
                self.encoding_list.append(encode_dict)
                self.decoding_list.append(decode_dict)
        if save_mode:
            self._save_encoder(path_to_save, dictionary_name)

    def encode(self, path_to_save: str = "./", **kwargs) -> DataFrame:
        """Encodes the `object` type columns of the dataframe

        Keyword Arguments:
        ----------
        - save_mode (`bool`): An optional integer parameter. By default it is set to `True`
        - dictionary_name (`str`): An optional string parameter. By default it is set to `labelencoder_dictionary`
        - norm_method (`str`): An optional string parameter to perform normalization. By default it is set to `None`
        """
        if len(self.encoding_list) == 0:
            self.train(path_to_save, **kwargs)
            return self._df

        else:
            print("Configuration detected")
            if len(self.median_list) == len(self._encode_columns):
                median_mode = True
            else:
                median_mode = False
            for num, colname in enumerate(self._encode_columns):
                if self._df[colname].dtype == "object":
                    encode_dict = self.encoding_list[num]
                    self._df[colname] = self._df[colname].apply(
                        self._code_transformation_to, dictionary_list=encode_dict
                    )
                    if median_mode:
                        self._df[colname] = self._df[colname].astype("float64")
                        self._df[colname] = self._df[colname] / self.median_list[num]
            return self._df

    def decode(self) -> DataFrame:
        """Decodes the `int` type columns of the `DataFrame`"""
        j = 0
        df_decoded = self._df.copy()
        if len(self.median_list) == len(self._encode_columns):
            median_mode = True
        else:
            median_mode = False
        try:
            number_of_columns = len(self.decoding_list[j])
            for i in self._encode_columns:
                if df_decoded[i].dtype == "int64" or df_decoded[i].dtype == "float64":
                    if median_mode:
                        df_decoded[i] = df_decoded[i] * self.median_list[j]
                        df_decoded[i] = df_decoded[i].astype("int64")
                    df_decoded[i] = df_decoded[i].apply(
                        self._code_transformation_to, dictionary_list=self.decoding_list[j]
                    )
                    j += 1
            return df_decoded
        except AttributeError as e:
            warning_type = "UserWarning"
            msg = "It is not possible to decode the dataframe, since it has not been encoded"
            msg += "Error: {%s}" % e
            print(f"{warning_type}: {msg}")

    def get_dictionaries(self) -> Tuple[List[dict], List[dict]]:
        """Allows to return the `list` of dictionaries for `encoding` and `decoding`"""
        try:
            return self.encoding_list, self.decoding_list
        except ValueError as e:
            warning_type = "UserWarning"
            msg = "It is not possible to return the list of dictionaries as they have not been created."
            msg += "Error: {%s}" % e
            print(f"{warning_type}: {msg}")

    def _save_encoder(self, path_to_save: str, dictionary_name: str) -> None:
        """Method to serialize the `encoding_list`, `decoding_list` and `_encode_columns` list"""
        with open(path_to_save + dictionary_name + ".pkl", "wb") as f:
            pickle.dump(
                [self.encoding_list, self.decoding_list, self._encode_columns, self.median_list], f
            )

    def _code_transformation_to(self, character: str, dictionary_list: List[dict]) -> int:
        """Auxiliary function to perform data transformation using a dictionary

        Parameters
        ----------
        character : `str`
            A character data type.
        dictionary_list : List[`dict`]
            An object of dictionary type.

        Returns
        -------
        dict_type[`character`] or `np.nan` if dict_type[`character`] doesn't exist.
        """
        try:
            return dictionary_list[character]
        except:
            return np.nan


class PerformanceMeasures:
    """Class with methods to measure performance"""

    def __init__(self) -> None:
        pass

    # Performance measure Res_T
    def f_mean(self, y_true: np.ndarray, y_pred: np.ndarray, labels: List[int]) -> float:
        F_vec = self._f1_score(y_true, y_pred, labels)
        mean_f_measure = np.mean(F_vec)

        for label, f_measure in zip(labels, F_vec):
            print(f"F-measure of label {label} -> {f_measure}")

        print(f"Mean of F-measure -> {mean_f_measure}")

        return mean_f_measure

    # Performance measure Res_P
    def resp(self, y_true: np.ndarray, y_pred: np.ndarray, labels: List[int]) -> float:
        T_C = len(y_true)
        sum1, sum2 = 0.0, 0.0
        F_vec = self._f1_score(y_true, y_pred, labels)

        for label_idx, label in enumerate(labels):
            class_instances = np.sum(y_true == label) / T_C
            sum1 += (1 - class_instances) * F_vec[label_idx]
            sum2 += 1 - class_instances

        res_p = sum1 / sum2 if sum2 != 0 else 0.0  # Avoid division by zero
        print(f"Metric Res_p -> {res_p}")

        return res_p

    def _summary_pred(self, y_true: np.ndarray, y_pred: np.ndarray, labels: List[int]) -> None:
        count_mat = self._confu_mat(y_true, y_pred, labels)
        print("       ", " | ".join(f"--{label}--" for label in labels))
        for i, label_i in enumerate(labels):
            row = [f"  {int(count_mat[i, j])}  " for j in range(len(labels))]
            print(f"--{label_i}--|", " | ".join(row))

    def _f1_score(self, y_true: np.ndarray, y_pred: np.ndarray, labels: List[int]) -> np.ndarray:
        count_mat = self._confu_mat(y_true, y_pred, labels)
        sum_cols = np.sum(count_mat, axis=0)
        sum_rows = np.sum(count_mat, axis=1)

        # Avoid division by zero
        precision = np.divide(
            count_mat.diagonal(), sum_cols, out=np.zeros_like(sum_cols), where=sum_cols != 0
        )
        recall = np.divide(
            count_mat.diagonal(), sum_rows, out=np.zeros_like(sum_rows), where=sum_rows != 0
        )
        f1_vec = 2 * ((precision * recall) / (precision + recall))

        return f1_vec

    # Returns confusion matrix of predictions
    def _confu_mat(self, y_true: np.ndarray, y_pred: np.ndarray, labels: List[int]) -> np.ndarray:
        num_classes = len(labels)
        label_mapping = {label: idx for idx, label in enumerate(labels)}
        count_mat = np.zeros((num_classes, num_classes))

        for pred_label, true_label in zip(y_pred, y_true):
            if pred_label in label_mapping and true_label in label_mapping:
                count_mat[label_mapping[pred_label], label_mapping[true_label]] += 1

        return count_mat


class OneHotEncoder:
    """
    Class used to encode categorical variables.
    It receives an array of integers and returns a binary array using the one-hot encoding method.
    """

    __slots__ = ["x"]

    def __init__(self) -> None:
        pass

    def encode(self, x: np.ndarray | list):
        self.x = x

        if not isinstance(self.x, np.ndarray):
            self.x = np.array(self.x)  # If not numpy array then convert it

        y = np.zeros(
            (self.x.size, self.x.max() + 1)
        )  # Build matrix of (size num of entries) x (max value + 1)

        y[np.arange(self.x.size), self.x] = 1  # Label with ones

        return y

    def decode(self, x: np.ndarray | list) -> np.ndarray:
        if not isinstance(x, np.ndarray):
            x = np.array(x)  # If not numpy array then convert it

        # We return the max values of each row
        y = np.argmax(x, axis=1)

        return y


class FeatureSelection:
    """
    Generate the data graph using a variation of the feature selection algorithm.

    - The method `get_digraph` returns the network based on the feature selection method.
    """

    __slots__ = ["not_features", "X", "all_features_imp_graph", "w_dict", "scaler"]

    def __init__(self, not_features: list[str] = []) -> None:
        """The initializer of the class. The initial parameter is a list of strings with variables to discard."""
        self.not_features: List[str] = not_features
        self.all_features_imp_graph: List[Tuple] = []
        self.w_dict = dict()

    def get_digraph(self, dataset: DataFrame, n_importances: int, use_scaler: bool = False) -> str:
        """
        Get directed graph showing importance of features.

        Parameters
        ----------
        dataset : `DataFrame`
            Dataset to be used for generating the graph.
        n_importances : `int`
            Number of top importances to show in the graph.

        Returns
        -------
        `str`
            A string representation of the directed graph.
        """
        # Assign and clean dataset
        self._load_data(dataset)

        curr_dataset = self.X
        columns = list(curr_dataset.columns)

        # We construct string from causal_graph
        feature_string = " digraph { "
        for column in columns:
            feature_string += column + "; "

        numeric_df = curr_dataset.select_dtypes(include="number")
        if use_scaler:
            self.scaler = DataScaler(numeric_df.copy().to_numpy().T, n=None)
            numeric_scaled = self.scaler.rescale()
            numeric_df = pd.DataFrame(numeric_scaled.T, columns=numeric_df.columns)
            curr_dataset[numeric_df.columns] = numeric_df

        # We construct dictionary to save index for scaling
        numeric_dict = dict(zip(list(numeric_df.columns), range(len(list(numeric_df.columns)))))

        # Iterate over all the columns to obtain their importances.
        for index_column, column in enumerate(columns):

            # Variable to predict
            Y = curr_dataset[column]

            # We check whether it is numerical or categorical.
            column_type = Y.dtype
            if column_type != "object":
                # Linear regression model
                Model = LinearRegression()

                # Auxiliary dataset without the column in question
                X_aux = curr_dataset.drop([column], axis=1)

                # We encode
                dfe = DataFrameEncoder(X_aux)
                encoded_df = dfe.encode(save_mode=False)
                # We train
                Model.fit(encoded_df.to_numpy().T, Y.to_numpy().T)
                # We obtain importance
                importance = Model.get_importances()
                w = Model.w
            else:
                Model = LogisticRegression()
                num_unique_entries = curr_dataset[column].nunique()

                quick_encoder = DataFrameEncoder(Y.to_frame())
                encoded_Y = quick_encoder.encode(save_mode=False)

                # Mapping to one-hot
                one_hot = OneHotEncoder()
                train_y = one_hot.encode(encoded_Y[column])
                # PASSING 0 -> 0.5 and 1 -> 0.73105
                for i in range(len(train_y)):
                    for j in range(num_unique_entries):
                        if train_y[i][j] == 1.0:
                            train_y[i][j] = 0.73105
                        else:
                            train_y[i][j] = 0.5

                # Delete the column in question
                X_aux = curr_dataset.drop([column], axis=1)

                # We encode
                dfe = DataFrameEncoder(X_aux)
                encoded_df = dfe.encode(save_mode=False)

                # We train
                Model.fit(encoded_df.to_numpy().T, train_y)

                # We obtain importance
                importance = Model.get_importances()
                w = Model.w

            # We obtain the $n$ most important ones
            top_n_indexes = sorted(
                range(len(importance)), key=lambda i: importance[i], reverse=True
            )[:n_importances]

            # We build the string for the column in question
            names_cols = list(X_aux.columns)
            # We store the indices, values and column names in a list of tuples.
            features_imp_node = [
                (names_cols[top_n_indexes[i]], importance[top_n_indexes[i]])
                for i in range(n_importances)
            ]
            # We store w's for predictions

            if column_type != "object":
                self.w_dict[column] = (w, None, names_cols, dfe, numeric_dict)
            else:
                self.w_dict[column] = (w, quick_encoder, names_cols, dfe, numeric_dict)
            # Add to general list
            self.all_features_imp_graph.append((column, features_imp_node))
            # We format it
            for i in top_n_indexes:
                feature_string += names_cols[i] + " -> "

            feature_string += column + "; "

        return feature_string + "} "

    def _load_data(self, dataset: DataFrame):
        # Assign data and clean dataset of unneeded columns

        if len(self.not_features) > 0:
            # We remove unnecessary columns
            self.X = dataset.drop(columns=self.not_features)

        else:
            self.X = dataset

        self.X.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.X.replace(" ", np.nan, inplace=True)
        self.X.dropna(inplace=True)
        self.X = self.X.reset_index()
        self.X = self.X.drop(columns=["index"])


def check_nan_inf(df: DataFrame) -> DataFrame:
    """
    Checks for NaN and Inf values in the DataFrame. If any are found, they will be removed.

    Parameters:
        df (DataFrame): The input DataFrame to be checked.

    Returns:
        DataFrame: A new DataFrame with NaN and Inf values removed.
    """

    nan_values = df.isnull().values.any()
    inf_values = np.isinf(df.select_dtypes(include="number")).values.any()

    if nan_values:
        print("UserWarning: Some rows may have been deleted due to the existence of NaN values.")
        df.dropna(inplace=True)

    if inf_values:
        print("UserWarning: Some rows may have been deleted due to the existence of Inf values.")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

    nan_count = df.isnull().values.sum()
    inf_count = np.isinf(df.select_dtypes(include="number")).values.sum()

    print(f"NaN values removed: {nan_count}")
    print(f"Infinite values removed: {inf_count}")

    return df


# -------------------------------------------------------------------------
if __name__ == "__main__":
    y_true = np.array([1, 2, 2, 1, 1])
    y_pred = np.array([1, 1, 2, 2, 1])

    labels = [1, 2]
    helper = PerformanceMeasures()
    helper._summary_pred(y_true, y_pred, labels)
    print(helper._f1_score(y_true, y_pred, labels))
    print(helper.f_mean(y_true, y_pred, labels))

    # Use DataFrameEncoder
    # Create a DataFrame
    data = {"Name": ["John", "Alice", "Bob", "Jafet", "Beto"], "Age": [25, 30, 35, 21, 28]}
    import pandas as pd

    df = pd.DataFrame(data)
    # Instantiate DataFrameEncoder
    dfe = DataFrameEncoder(df)
    # Encode the dataframe
    encoded_df = dfe.encode(norm_method="median")
    # Decode the dataframe
    decoded_df = dfe.decode()

    # Instantiate DataFrameEncoder
    # Use load_config method
    dfe2 = DataFrameEncoder(df)
    dfe2.load_config()

    encoded_df2 = dfe2.encode()
    # Decode the dataframe
    decoded_df2 = dfe2.decode()
    # Check if the loaded dictionaries match the original ones
    assert dfe.encoding_list == dfe2.encoding_list
    assert dfe.decoding_list == dfe2.decoding_list

    # Generate data
    x = np.random.rand(3, 100)
    y = 0.1 * x[0, :] + 0.4 * x[1, :] + 0.5 * x[2, :] + 0.1
    # Create a DataFrame
    df = pd.DataFrame(x.T, columns=["x1", "x2", "x3"])
    df["y"] = y
    # Instantiate FeatureSelection
    fs = FeatureSelection()
    print(fs.get_digraph(df, n_importances=1))

    linear_model = LinearRegression()
    linear_model.fit(x, y)
    importance = linear_model.get_importances()
    y_hat = linear_model.predict(x)

    # Graph the data for visualization
    plt.plot(x[0, :], y, "o", label="Original Data")
    plt.plot(x[0, :], y_hat, "x", label="$\hat{y}$")
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$y, $\hat{y}$")
    plt.show()

    a = generate_series(1, 40, incline=False)
    # Graph the data for visualization
    plt.plot(range(len(a[0, :])), a[0, :], label="Original Data")
    plt.legend()
    plt.xlabel("Time periods")
    plt.ylabel("$y(t)$")
    plt.show()

    a_denoise, _ = fft_denoise(a)

    plt.plot(range(len(a_denoise[0, :])), a_denoise[0, :], label="Denoise Data")
    plt.legend()
    plt.xlabel("Time periods")
    plt.ylabel("$y(t)$")
    plt.show()

    # Calculate the autocorrelation of the data
    z = AutoCorrelation(a[0, :])
    z.plot()
    # print(z())

    N = 1000
    mu = np.random.uniform(0, 10.0)
    sigma = np.random.uniform(0.1, 1.0)
    x = np.random.normal(mu, sigma, N)
    f, cdf_, ox = cdf(x, plot=True)
    invf, cdf_, ox = cdf(x, plot=True, inv=True)

    encoder = OneHotEncoder()
    encoding = encoder.encode([1, 2, 3, 4, 5])
    assert np.array_equal(
        encoding,
        np.array(
            [
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        ),
    )
