from typing import Callable, List, Tuple

import corner
import matplotlib.pyplot as plt
import numpy as np


def lnprior(theta: np.ndarray, conditions: List[Tuple[float, float]]) -> float:
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
    if len(conditions) != 2 * len(theta):
        raise ValueError("Length of conditions must be twice the length of theta.")

    cond = np.array(conditions).reshape((len(theta), 2))
    within_bounds = np.logical_and(cond[:, 0] < theta, theta < cond[:, 1])
    if not np.all(within_bounds):
        return np.inf

    return 0.0


def fun_like(
    x: np.ndarray,
    y: np.ndarray,
    model: Callable,
    theta: np.ndarray,
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
        A Python function defined by the user. This function should receive
        two arguments $(x, theta)$.
    theta : `np.ndarray`
        The array containing the model's parameters.
    conditions : `list`, optional
        A list containing $2n$-conditions for the (min, max) range of the
        $n$ parameters. Defaults to None.
    var2 : `float`, optional
        Determines the step size of the walker. By default it is set to `1.0`.

    Returns
    -------
    lhood : `float`
        The computed likelihood.
    """
    lp = 0.0 if conditions is None else lnprior(theta, conditions)
    inv_sigma2 = 1.0 / var2
    y_hat = model(x, theta)

    try:
        y_hat.shape[1]
    except IndexError:
        y_hat = y_hat[np.newaxis, ...].T

    y_sum = np.sum((y - y_hat) ** 2 * inv_sigma2 - np.log(inv_sigma2))
    lhood = 0.5 * y_sum

    if not np.isfinite(lhood):
        return np.inf

    return lhood + lp


def update_theta(theta: np.ndarray, d: float) -> np.ndarray:
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
    return np.random.normal(theta, d / 2.0)


def walk(
    x: np.ndarray,
    y: np.ndarray,
    model: Callable,
    theta: np.ndarray,
    conditions: List[Tuple[float, float]] = None,
    var2: float = 0.01,
    mov: int = 100,
    d: float = 1.0,
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
        A Python function defined by the user. This function should receive
        two arguments $(x, theta)$.
    theta : `np.ndarray`
        The array containing the model's parameters.
    conditions : `list`, optional
        A list containing $2n$-conditions for the (min, max) range of the
        $n$ parameters. Defaults to None.
    var2 : `float`, optional
        Determines the step size of the walker. By default it is set to `1.0`.
    mov : `int`, optional
        Number of movements that walker will perform. By default it is set
        to `100`.
    d : `float`, optional
        Size of the Gaussian step for the walker.
    tol : `float`, optional
        Convergence criteria for the log-likelihood. By default it is set
        to `1e-3`.
    mode : `bool`, optional
        Defaults to `True`.

    Returns
    -------
    theta : `np.array`
        An ndarray with the updated theta values.
    nwalk : `np.array`
        Updates of theta for each movement performed by the walker.
    y0 : `float`
        The log-likelihood value.
    """
    nwalk = []

    for i in range(mov):
        nwalk.append(theta)
        theta_new = update_theta(theta, d)

        y0 = fun_like(x, y, model, theta, conditions, var2)
        y1 = fun_like(x, y, model, theta_new, conditions, var2)
        if y0 <= tol or y1 <= tol:
            if mode:
                print("Goal reached!")
            return (theta_new, nwalk, y1) if y1 <= tol else (theta, nwalk, y0)

        if y1 >= y0:
            ratio = y0 / y1
            prob = np.exp(-ratio)

            if prob > np.random.rand():
                theta = theta_new
        else:
            theta = theta_new
            theta_new = update_theta(theta, d)

    if mode:
        print("Maximum number of iterations reached!")
        print(f"The log-likelihood is: {y0}")

    return theta, nwalk, y0


def walkers(
    nwalkers: int,
    x: np.ndarray,
    y: np.ndarray,
    model: Callable,
    theta: np.ndarray,
    conditions: List[Tuple[float, float]] = None,
    var2: float = 0.01,
    mov: int = 100,
    d: float = 1.0,
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
        A Python function defined by the user. This function should receive
        two arguments $(x, theta)$.
    theta : `np.ndarray`
        The array containing the model's parameters.
    conditions : `list`, optional
        A list containing $2n$-conditions for the (min, max) range of the
        $n$ parameters. Defaults to None.
    var2 : `float`, optional
        Determines the step size of the walker. By default it is set to `1.0`.
    mov : `int`, optional
        Number of movements that walker will perform. By default it is set
        to `100`.
    d : `float`, optional
        Size of the Gaussian step for the walker.
    tol : `float`, optional
        Convergence criteria for the log-likelihood. By default it is set
        to `1e-3`.
    mode : `bool`, optional
        Specifies that we will be working with more than one walker. By
        default it is set to `False`.
    figname : `str`, optional
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

        if figname:
            for k in range(nwalk.shape[1]):
                sub = f"$\\theta _{k}$"
                plt.plot(range(len(nwalk[:, k])), nwalk[:, k], "-", label=sub)
                plt.ylabel("$\\theta$")
                plt.xlabel("iterations")
                plt.savefig(f"walkers_{figname}", dpi=300, transparent=True)

    if figname:
        plt.show()

    if len(theta) == 2 and figname:
        corner.hist2d(
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
        plt.savefig(f"theta_{figname}", dpi=300, transparent=True)

    return par, error
