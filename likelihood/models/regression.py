import pickle

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

from likelihood.main import *
from likelihood.models.utils import FeaturesArima
from likelihood.tools import *

# -------------------------------------------------------------------------


class AbstractArima(FeaturesArima):
    """A class that implements the auto-regressive arima (1, 0, 0) model"""

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

    def __init__(self, datapoints: ndarray, n_steps: int = 0, noise: float = 0, tol: float = 1e-4):
        self.datapoints = datapoints
        self.n_steps = n_steps
        self.noise = noise
        self.p = datapoints.shape[0]
        self.q = 0
        self.tol = tol

    def model(self, datapoints: ndarray, theta: list, mode=True):
        datapoints = self.datapoints
        noise = self.noise
        self.theta_trained = theta

        return super().forward(datapoints, theta, mode, noise)

    def xvec(self, datapoints: ndarray, n_steps: int = 0):
        datapoints = self.datapoints
        self.n_steps = n_steps

        return datapoints[n_steps:]

    def train(self, nwalkers: int = 1, mov: int = 200, weights: bool = False):
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

    def eval(self, y_val: ndarray, y_pred: ndarray):
        rmse = np.sqrt(np.mean((y_pred - y_val) ** 2))
        square_error = np.sqrt((y_pred - y_val) ** 2)
        accuracy = np.sum(square_error[np.where(square_error < rmse)])
        accuracy /= np.sum(square_error)
        print("Accuracy: {:.4f}".format(accuracy))
        print("RMSE: {:.4f}".format(rmse))

    def plot_pred(self, y_real: ndarray, y_pred: ndarray, ci: float = 0.90, mode: bool = True):
        plt.figure()
        n = self.n_steps
        y_mean = np.mean(y_pred, axis=0)
        y_std = np.std(y_pred, axis=0)
        if ci < 0.95:
            Z = (ci / 0.90) * 1.64
        else:
            Z = (ci / 0.95) * 1.96

        plt.plot(y_pred, label="Predicted")
        plt.plot(y_real, ".--", label="Real", alpha=0.5)
        plt.fill_between(
            (range(y_pred.shape[0]))[-n:],
            (y_pred - Z * y_std)[-n:],
            (y_pred + Z * y_std)[-n:],
            alpha=0.2,
        )
        plt.xlabel("Time steps")
        plt.ylabel("y")
        plt.legend()
        print("Confidence Interval: {:.4f}".format(Z * y_std))
        if mode:
            plt.savefig("pred_" + str(n) + ".png", dpi=300)
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

    datapoints : np.array
        A set of points to train the arima model.

    n_steps : int
        Is the number of points that in predict(n_steps)
        stage will estimate foward. By default it is set to `0`.

    Returns
    -------

    new_datapoints : np.array
        It is the number of predicted points. It is necessary
        to apply predict(n_steps) followed by fit()

    """

    __slots__ = ["datapoints_", "n_steps", "sigma", "mode", "mov", "n_walkers", "name"]

    def __init__(self, datapoints: ndarray, n_steps: int = 0):
        self.datapoints_ = datapoints
        self.n_steps = n_steps

    def fit(self, sigma: int = 0, mov: int = 200, mode: bool = False):
        self.sigma = sigma
        self.mode = mode
        self.mov = mov

        datapoints = self.datapoints_
        self.datapoints_, _ = fft_denoise(datapoints, sigma, mode)

    def predict(
        self, n_steps: int = 0, n_walkers: int = 1, name: str = "fourier_model", save: bool = True
    ):
        self.n_steps = n_steps
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

    def load_predict(self, name: str = "fourier_model"):
        n_steps = self.n_steps

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
    """A class that implements the (p, d, q) arima model

    Parameters
    ----------

    datapoints : np.array
        A set of points to train the arima model.

    p : float
        Is the number of auto-regressive terms (ratio). By default it is set to `1`

    d : int
        Is known as the degree of differencing. By default it is set to `0`

    q : float
        Is the number of forecast errors in the model (ratio). By default it is set to `0`

    Returns
    -------

    y_pred : np.array
        It is the number of predicted points. It is necessary
        to apply predict(n_steps) followed by train()
    """

    __slots__ = ["datapoints", "n_steps", "noise", "p", "d", "q", "tol", "theta_trained"]

    def __init__(
        self,
        datapoints: ndarray,
        p: float = 1,
        d: int = 0,
        q: float = 0,
        n_steps: int = 0,
        noise: float = 0,
        tol: float = 1e-5,
    ):
        self.datapoints = datapoints
        self.n_steps = n_steps
        self.noise = noise
        assert p > 0 and p <= 1, "p must be less than 1 but greater than 0"
        self.p = int(p * len(datapoints))
        assert d >= 0 and d <= 1, "p must be less than 1 but greater than or equal to 0"
        self.d = d
        self.q = int(q * len(datapoints))
        self.tol = tol

    def model(self, datapoints: ndarray, theta: list, mode: bool = True):
        datapoints = self.datapoints
        noise = self.noise
        self.theta_trained = theta

        assert type(self.d) == int, "d must be 0, 1 or 2"

        if self.d != 0 or self.q != 0:
            if self.d != 0:
                y_sum = super().integrated(datapoints)
            else:
                y_sum = datapoints

            y_sum_regr = y_sum[-self.p :]
            y_regr_vec = super().forward(y_sum_regr, theta[0 : self.p], mode, 0)
            if self.q != 0:
                y_sum_average = super().average(y_sum[-self.q :])
                y_average_vec = super().forward(y_sum_average, theta[-self.q :], mode, 0)
                if mode:
                    y_vec = y_regr_vec
                    for i in reversed(range(y_average_vec.shape[0])):
                        y_vec[i] += y_average_vec[i]
                else:
                    y_vec = y_regr_vec + y_average_vec
            else:
                y_vec = y_regr_vec

            return y_vec
        else:
            return super().forward(datapoints, theta, mode, noise)
