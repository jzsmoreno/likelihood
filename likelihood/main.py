from typing import Callable, List, Tuple

import corner
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray


def lnprior(theta: ndarray, conditions: List[Tuple[float, float]]) -> float:
    """Computes the prior probability.

    Parameters
    ----------
    theta : `np.ndarray`
        An array containing the parameters of the model.
    conditions : `list`
        A list containing $2n$-conditions for the (min, max) range of the
        $n$ parameters.

    Returns
    -------
    lp : `float`
        The a priori probability.
    """

    try:
        if len(conditions) != 2 * len(theta):
            error_type = "IndexError"
            msg = "Length of conditions must be twice the length of theta."
            print(f"{error_type}: {msg}")
        else:
            cond = np.array(conditions).reshape((len(theta), 2))
            for i in range(len(theta)):
                if cond[i, 0] < theta[i] < cond[i, 1]:
                    lp = 0.0
                else:
                    return np.inf
            return lp
    except:
        return 0.0


def fun_like(
    x: ndarray,
    y: ndarray,
    model: Callable,
    theta: ndarray,
    conditions: List[Tuple[float, float]] = None,
    var2: float = 1.0,
) -> float:
    """Computes the likelihood.

    Parameters
    ----------
    x : `np.ndarray`
        An $(m, n)$ dimensional array for (cols, rows).
    y : `np.ndarray`
        An $n$ dimensional array that will be compared with model's output.
    model : `Callable`
        A Python function defined by the user. This function should recieve
        two arguments $(`x`, `theta`)$.
    theta : `np.ndarray`
        The array containing the model's parameters.
    conditions : `list`
        A list containing $2n$-conditions for the (min, max) range of the
        $n$ parameters.
    var2 : `float`
        Determines the step size of the walker. By default it is set to `1.0`.

    Returns
    -------
    lhood : `float`
        The computed likelihood.
    """

    lp = lnprior(theta, conditions)
    inv_sigma2 = 1.0 / (var2)
    y_hat = model(x, theta)

    try:
        y_hat.shape[1]
    except:
        y_hat = y_hat[np.newaxis, ...].T

    y_sum = np.sum((y - y_hat) ** 2 * inv_sigma2 - np.log(inv_sigma2))
    lhood = 0.5 * y_sum

    if not np.isfinite(lp):
        lhood = np.inf
    else:
        lhood += lp

    return lhood


def update_theta(theta: ndarray, d: float) -> ndarray:
    """Updates the theta parameters.

    Parameters
    ----------
    theta : `np.ndarray`
        The ndarray containing the model's parameters.
    d : `float`
        Size of the Gaussian step for the walker.

    Returns
    -------
    theta_new : `np.array`
        An ndarray with the updated theta values.
    """

    theta_new = [np.random.normal(theta[k], d / 2.0) for k in range(len(theta))]

    return theta_new


def walk(
    x: ndarray,
    y: ndarray,
    model: Callable,
    theta: ndarray,
    conditions: List[Tuple[float, float]] = None,
    var2: float = 0.01,
    mov: int = 100,
    d: int = 1,
    tol: float = 1e-4,
    mode: bool = True,
):
    """Executes the walker implementation.

    Parameters
    ----------
    x : `np.ndarray`
        An $(m, n)$ dimensional array for (cols, rows).
    y : np.ndarray
        An $n$ dimensional array that will be compared with model's output.
    model : `Callable`
        A Python function defined by the user. This function should recieve
        two arguments $(x, theta)$.
    theta : `np.ndarray`
        The array containing the model's parameters.
    conditions : `list`
        A list containing $2n$-conditions for the (min, max) range of the
        $n$ parameters.
    var2 : `float`
        Determines the step size of the walker. By default it is set to `1.0`.
    mov : `int`
        Number of movements that walker will perform. By default it is set
        to `100`.
    d : `float`
        Size of the Gaussian step for the walker.
    tol : `float`
        Convergence criteria for the log-likelihood. By default it is set
        to `1e-3`.
    mode : `bool`
        By default it is set to `True`.

    Returns
    -------
    theta : `np.array`
        An ndarray with the updated theta values.
    nwalk : `np.array`
        Updates of theta for each movement performed by the walker.
    y0 : `float`
        The log-likelihood value.
    """

    greach = False
    nwalk = []

    for i in range(mov):
        nwalk.append(theta)
        theta_new = update_theta(theta, d)

        if not greach:
            y0 = fun_like(x, y, model, theta, conditions, var2)
            y1 = fun_like(x, y, model, theta_new, conditions, var2)

            if y0 <= tol and mode:
                print("Goal reached!")
                greach = True

                return theta, nwalk, y0
            else:
                if y1 <= tol and mode:
                    print("Goal reached!")
                    greach = True

                    return theta_new, nwalk, y1
                else:
                    ratio = y0 / y1
                    boltz = np.random.rand(1)
                    prob = np.exp(-ratio)

                    if y1 < y0:
                        theta = theta_new
                        theta_new = update_theta(theta, d)
                    else:
                        if prob > boltz:
                            theta = theta_new
                            theta_new = update_theta(theta, d)
                        else:
                            theta_new = update_theta(theta, d)
    if mode:
        print("Maximum number of iterations reached!")
        print(f"The log-likelihood is: {y0}")

    return theta, nwalk, y0


def walkers(
    nwalkers: int,
    x: ndarray,
    y: ndarray,
    model: Callable,
    theta: ndarray,
    conditions: bool = None,
    var2: float = 0.01,
    mov: int = 100,
    d: int = 1,
    tol: float = 1e-4,
    mode: bool = False,
    figname: str = "fig_out.png",
):
    """Executes multiple walkers.

    Parameters
    ----------
    nwalkers : `int`
        The number of walkers to be executed.
    x : `np.ndarray`
        An $(m, n)$ dimensional array for (cols, rows).
    y : `np.ndarray`
        An $n$ dimensional array that will be compared with model's output.
    model : `Callable`
        A Python function defined by the user. This function should recieve
        two arguments $(x, theta)$.
    theta : `np.ndarray`
        The array containing the model's parameters.
    conditions : `list`
        A list containing $2n$-conditions for the (min, max) range of the
        $n$ parameters.
    var2 : `float`
        Determines the step size of the walker. By default it is set to `1.0`.
    mov : `int`
        Number of movements that walker will perform. By default it is set
        to `100`.
    d : `float`
        Size of the Gaussian step for the walker.
    tol : `float`
        Convergence criteria for the log-likelihhod. By default it is set
        to `1e-3`.
    mode : `bool`
        Specifies that we will be working with more than one walker. By
        default it is set to `False`.
    figname : `str`
        The name of the output file for the figure. By default it is set
        to `fig_out.png`.

    Returns
    -------
    par : `np.array`
        The theta found by each of the walkers.
    error : `np.array`
        The log-likelihood array.
    """

    error = []
    par = []

    for i in range(nwalkers):
        theta, nwalk, y0 = walk(x, y, model, theta, conditions, var2, mov, d, tol, mode)
        par.append(theta)
        nwalk = np.array(nwalk).reshape((len(nwalk), len(nwalk[i])))
        error.append(y0)

        if figname != None:
            for k in range(nwalk.shape[1]):
                sub = "$\\theta _{" + str(k) + "}$"
                plt.plot(range(len(nwalk[:, k])), nwalk[:, k], "-", label=sub)
                plt.ylabel("$\\theta$")
                plt.xlabel("iterations")
                plt.savefig("walkers_" + figname, dpi=300, transparent=True)

    if figname != None:
        plt.show()

    if nwalk.shape[1] == 2:
        if figname != None:
            fig = corner.hist2d(
                nwalk[:, 0],
                nwalk[:, 1],
                range=None,
                bins=18,
                smooth=True,
                plot_datapoints=True,
                plot_density=True,
            )
            plt.ylabel("$\\theta_{1}$")
            plt.xlabel("$\\theta_{0}$")
            plt.savefig("theta_" + figname, dpi=300, transparent=True)

    return par, error
