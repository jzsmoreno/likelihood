import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from likelihood.main import *
from likelihood.models.utils import FeaturesArima
from likelihood.tools import *

# -------------------------------------------------------------------------


class AbstractArima(FeaturesArima):
    """A class that implements the auto-regressive arima (1, 0, 0) model

    Parameters
    ----------
    datapoints : `np.ndarray`
        The input data points for training.
    noise : `float`, optional
        Noise level for the model, by default 0
    tol : `float`, optional
        Tolerance for convergence, by default 1e-4

    Attributes
    ----------
    datapoints : `np.ndarray`
        The input data points for training.
    n_steps : `int`
        Number of steps to predict.
    noise : `float`
        Noise level for the model.
    p : `int`
        Order of autoregressive part.
    q : `int`
        Order of moving average part.
    tol : `float`
        Tolerance for convergence.
    nwalkers : `int`
        Number of walkers for sampling.
    mov : `int`
        Maximum number of iterations.
    theta_trained : `np.ndarray`
        Trained parameters of the model.
    """

    __slots__ = [
        "datapoints",
        "n_steps",
        "noise",
        "p",
        "q",
        "tol",
        "nwalkers",
        "mov",
        "theta_trained",
    ]

    def __init__(self, datapoints: np.ndarray, noise: float = 0, tol: float = 1e-4):
        """Initialize the ARIMA model.

        Parameters
        ----------
        datapoints : `np.ndarray`
            The input data points for training.
        noise : `float`, optional
            Noise level for the model, by default 0
        tol : `float`, optional
            Tolerance for convergence, by default 1e-4
        """
        self.datapoints = datapoints
        self.noise = noise
        self.p = datapoints.shape[0]
        self.q = 0
        self.tol = tol
        self.n_steps = 0

    def model(self, datapoints: np.ndarray, theta: list, mode=True):
        """Compute the model forward pass.

        Parameters
        ----------
        datapoints : `np.ndarray`
            The input data points.
        theta : `list`
            Model parameters.
        mode : `bool`, optional
            Forward pass mode, by default True

        Returns
        -------
        `np.ndarray`
            Model output.
        """
        datapoints = self.datapoints
        noise = self.noise
        self.theta_trained = theta

        return super().forward(datapoints, theta, mode, noise)

    def xvec(self, datapoints: np.ndarray, n_steps: int = 0):
        """Extract vector of data points.

        Parameters
        ----------
        datapoints : `np.ndarray`
            The input data points.
        n_steps : `int`, optional
            Number of steps to consider, by default 0

        Returns
        -------
        `np.ndarray`
            Extracted data points vector.
        """
        datapoints = self.datapoints
        self.n_steps = n_steps

        return datapoints[n_steps:]

    def train(self, nwalkers: int = 10, mov: int = 200, weights: bool = False):
        """Train the model using sampling method.

        Parameters
        ----------
        nwalkers : `int`, optional
            Number of walkers for sampling, by default 10
        mov : `int`, optional
            Maximum number of iterations, by default 200
        weights : `bool`, optional
            Whether to use weights in sampling, by default False
        """
        datapoints = self.datapoints
        xvec = self.xvec
        self.nwalkers = nwalkers
        self.mov = mov

        assert self.nwalkers <= self.mov, "n_walkers must be less or equal than mov"
        model = self.model
        n = self.p + self.q
        theta = np.random.rand(n)
        x_vec = xvec(datapoints)

        if weights:
            par, error = walkers(
                nwalkers,
                x_vec,
                datapoints,
                model,
                theta=self.theta_trained,
                mov=mov,
                tol=self.tol,
                figname=None,
            )
        else:
            par, error = walkers(
                nwalkers, x_vec, datapoints, model, theta, mov=mov, tol=self.tol, figname=None
            )

        index = np.where(error == np.min(error))[0][0]
        trained = np.array(par[index])

        self.theta_trained = trained

    def predict(self, n_steps: int = 0):
        """Make predictions for future steps.

        Parameters
        ----------
        n_steps : `int`, optional
            Number of steps to predict, by default 0

        Returns
        -------
        `np.ndarray`
            Predicted values.
        """
        self.n_steps = n_steps
        datapoints = self.datapoints
        model = self.model
        theta_trained = self.theta_trained
        y_pred = model(datapoints, theta_trained)

        for i in range(n_steps):
            self.datapoints = y_pred[i:]
            y_new = model(datapoints, theta_trained, mode=False)
            y_pred = y_pred.tolist()
            y_pred.append(y_new)
            y_pred = np.array(y_pred)

        return np.array(y_pred)

    def save_model(self, name: str = "model"):
        with open(name + ".pkl", "wb") as file:
            pickle.dump(self.theta_trained, file)

    def load_model(self, name: str = "model"):
        with open(name + ".pkl", "rb") as file:
            self.theta_trained = pickle.load(file)

    def eval(self, y_val: np.ndarray, y_pred: np.ndarray):
        rmse = np.sqrt(np.mean((y_pred - y_val) ** 2))
        square_error = np.sqrt((y_pred - y_val) ** 2)
        accuracy = np.sum(square_error[np.where(square_error < rmse)])
        accuracy /= np.sum(square_error)
        print("Accuracy: {:.4f}".format(accuracy))
        print("RMSE: {:.4f}".format(rmse))

    def plot_pred(
        self, y_real: np.ndarray, y_pred: np.ndarray, ci: float = 0.90, mode: bool = True
    ):
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(5, 3))
        n = self.n_steps
        y_mean = np.mean(y_pred, axis=0)
        y_std = np.std(y_pred, axis=0)
        if ci < 0.95:
            Z = (ci / 0.90) * 1.64
        else:
            Z = (ci / 0.95) * 1.96
        plt.plot(y_pred, label="Predicted", linewidth=2, color=sns.color_palette("deep")[1])
        plt.plot(
            y_real, ".--", label="Real", color=sns.color_palette("deep")[0], alpha=0.6, markersize=6
        )
        plt.fill_between(
            range(y_pred.shape[0])[-n:],
            (y_pred - Z * y_std)[-n:],
            (y_pred + Z * y_std)[-n:],
            alpha=0.2,
            color=sns.color_palette("deep")[1],
        )
        plt.title("Predicted vs Real Values with Confidence Interval", fontsize=12)
        plt.xlabel("Time Steps", fontsize=12)
        plt.ylabel("y", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        print(f"Confidence Interval: ±{Z * y_std:.4f}")
        plt.legend(loc="upper left", fontsize=9)
        if mode:
            plt.savefig(f"pred_{n}.png", dpi=300)
        plt.tight_layout()
        plt.show()

    def summary(self):
        print("\nSummary:")
        print("-----------------------")
        print("Lenght of theta: {}".format(len(self.theta_trained)))
        print("Mean of theta: {:.4f}".format(np.mean(self.theta_trained)))
        print("-----------------------")


class FourierRegression(AbstractArima):
    """A class that implements the arima model with FFT noise filtering

    Parameters
    ----------
    datapoints : np.ndarray
        A set of points to train the arima model.

    Returns
    -------
    new_datapoints : np.ndarray
        It is the number of predicted points. It is necessary
        to apply predict(n_steps) followed by fit()

    """

    __slots__ = ["datapoints_", "sigma", "mode", "mov", "n_walkers", "name"]

    def __init__(self, datapoints: np.ndarray):
        self.datapoints_ = datapoints

    def fit(self, sigma: int = 0, mov: int = 200, mode: bool = False):
        self.sigma = sigma
        self.mode = mode
        self.mov = mov

        datapoints = self.datapoints_
        self.datapoints_, _ = fft_denoise(datapoints, sigma, mode)

    def predict(
        self, n_steps: int, n_walkers: int = 1, name: str = "fourier_model", save: bool = True
    ):
        self.n_walkers = n_walkers
        self.name = name
        mov = self.mov

        assert self.n_walkers <= mov, "n_walkers must be less or equal than mov"

        new_datapoints = []
        for i in range(self.datapoints_.shape[0]):
            super().__init__(self.datapoints_[i, :])
            super().train(n_walkers, mov)
            if save:
                super().save_model(str(i) + "_" + name)
            y_pred_ = super().predict(n_steps)
            new_datapoints.append(y_pred_)

        new_datapoints = np.array(new_datapoints)
        new_datapoints = np.reshape(new_datapoints, (len(new_datapoints), -1))

        return new_datapoints

    def load_predict(self, n_steps: int, name: str = "fourier_model"):
        new_datapoints = []

        for i in range(self.datapoints_.shape[0]):
            super().__init__(self.datapoints_[i, :])
            super().load_model(str(i) + "_" + name)
            y_pred_ = super().predict(n_steps)
            new_datapoints.append(y_pred_)

        new_datapoints = np.array(new_datapoints)
        new_datapoints = np.reshape(new_datapoints, (len(new_datapoints), -1))

        return new_datapoints


class Arima(AbstractArima):
    """A class that implements the (p, d, q) ARIMA model.

    Parameters
    ----------
    datapoints : np.ndarray
        A set of points to train the ARIMA model.
    p : float
        Number of auto-regressive terms (ratio). By default it is set to `1`.
    d : int
        Degree of differencing. By default it is set to `0`.
    q : float
        Number of forecast errors in the model (ratio). By default it is set to `0`.
    n_steps : int
        Number of steps to predict ahead.
    noise : float
        Amount of noise added during training.
    tol : float
        Tolerance for convergence checks.

    Returns
    -------
    None

    Notes
    -----
    The values of `p`, `q` are scaled based on the length of `datapoints`.
    """

    __slots__ = ["datapoints", "noise", "p", "d", "q", "tol", "theta_trained"]

    def __init__(
        self,
        datapoints: np.ndarray,
        p: float = 1,
        d: int = 0,
        q: float = 0,
        noise: float = 0,
        tol: float = 1e-5,
    ):
        """Initializes the ARIMA model with given parameters.

        Parameters
        ----------
        datapoints : np.ndarray
            A set of points to train the ARIMA model.
        p : float
            Auto-regressive term (scaled by length of data).
        d : int
            Degree of differencing.
        q : float
            Moving average term (scaled by length of data).
        noise : float
            Noise level for training.
        tol : float
            Tolerance for numerical convergence.

        Returns
        -------
        None
        """
        self.datapoints = datapoints
        self.noise = noise
        assert p > 0 and p <= 1, "p must be less than 1 but greater than 0"
        self.p = int(p * len(datapoints))
        assert d >= 0 and d <= 1, "p must be less than 1 but greater than or equal to 0"
        self.d = d
        self.q = int(q * len(datapoints))
        self.tol = tol

    def model(self, datapoints: np.ndarray, theta: list, mode: bool = True):
        """Computes the prior probability or prediction based on ARIMA model.

        Parameters
        ----------
        datapoints : np.ndarray
            The input data used for modeling.
        theta : list
            Model parameters.
        mode : bool
            If True, computes in forward mode; otherwise in backward mode.

        Returns
        -------
        y_vec : np.ndarray
            Predicted values according to the ARIMA model.
        """
        datapoints = self.datapoints
        noise = self.noise
        self.theta_trained = theta

        assert type(self.d) == int, "d must be 0 or 1"

        if self.d != 0 or self.q != 0:
            if self.d != 0:
                y_sum = super().integrated(datapoints)
                norm_datapoints = np.linalg.norm(datapoints)
                norm_y_sum = np.linalg.norm(y_sum)
                if norm_y_sum != 0 and norm_datapoints != 0:
                    y_sum = cal_average(
                        np.abs(y_sum * (norm_datapoints / norm_y_sum)) * np.sign(datapoints), 0.05
                    )
            else:
                y_sum = datapoints.copy()

            y_sum_regr = y_sum[-self.p :]
            y_regr_vec = super().forward(y_sum_regr, theta[0 : self.p], mode, 0)
            if self.q != 0:
                y_sum_average = super().average(y_sum[-self.q :])
                y_vec_magnitude = np.linalg.norm(y_regr_vec.copy())
                y_sum_average_magnitude = np.linalg.norm(y_sum_average)

                if y_sum_average_magnitude > y_vec_magnitude:
                    scaling_factor = y_vec_magnitude / y_sum_average_magnitude
                    y_sum_average = y_sum_average * scaling_factor
                theta_mean = np.mean(theta[-self.q :])
                if abs(theta_mean) > 1:
                    additional_scaling_factor = 1.0 - abs(theta_mean)
                    y_sum_average = y_sum_average * additional_scaling_factor
                y_average_vec = super().forward(y_sum_average, theta[-self.q :], mode, 0)
                if mode:
                    y_vec = y_regr_vec.copy()
                    for i in reversed(range(y_average_vec.shape[0])):
                        y_vec[i] += y_average_vec[i]
                else:
                    y_vec = y_regr_vec + y_average_vec
            else:
                y_vec = y_regr_vec
            return y_vec
        else:
            return super().forward(datapoints, theta, mode, noise)
