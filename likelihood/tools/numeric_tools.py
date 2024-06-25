from typing import Dict

import numpy as np
from numpy import arange, array, ndarray, random
from numpy.linalg import solve
from pandas.core.frame import DataFrame

# -------------------------------------------------------------------------


def xi_corr(df: DataFrame) -> DataFrame:
    """Calculate new coefficient of correlation for all pairs of columns in a `DataFrame`.

    Parameters
    ----------
    df : `DataFrame`
        Input data containing the variables to be correlated.

    Returns
    -------
    `DataFrame`
        A dataframe with variable names as keys and their corresponding
        correlation coefficients as values.
    """
    correlations = {}
    columns = df.columns

    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            if i < j:
                x = df[col1].values
                y = df[col2].values

                correlation = xicor(x, y)
                correlations[(col1, col2)] = round(correlation, 8)
    # dictionary to dataframe
    correlations = DataFrame(list(correlations.items()), columns=["Variables", "Xi Correlation"])
    return correlations


"""
@article{Chatterjee2019ANC,
  title={A New Coefficient of Correlation},
  author={Sourav Chatterjee},
  journal={Journal of the American Statistical Association},
  year={2019},
  volume={116},
  pages={2009 - 2022},
  url={https://api.semanticscholar.org/CorpusID:202719281}
} 
"""


def xicor(X: ndarray, Y: ndarray, ties: bool = True) -> float:
    """Calculate a new coefficient of correlation between two variables.

    The new coefficient of correlation is a generalization of Pearson's correlation.

    Parameters
    ----------
    X : `np.ndarray`
        The first variable to be correlated. Must have at least one dimension.
    Y : `np.ndarray`
        The second variable to be correlated. Must have at least one dimension.

    Returns
    -------
    xi : `float`
        The estimated value of the new coefficient of correlation.
    """
    random.seed(42)
    n = len(X)
    order = array([i[0] for i in sorted(enumerate(X), key=lambda x: x[1])])
    if ties:
        l = array([sum(y >= Y[order]) for y in Y[order]])
        r = l.copy()
        for j in range(n):
            if sum([r[j] == r[i] for i in range(n)]) > 1:
                tie_index = array([r[j] == r[i] for i in range(n)])
                r[tie_index] = random.choice(
                    r[tie_index] - arange(0, sum([r[j] == r[i] for i in range(n)])),
                    sum(tie_index),
                    replace=False,
                )
        return 1 - n * sum(abs(r[1:] - r[: n - 1])) / (2 * sum(l * (n - l)))
    else:
        r = array([sum(y >= Y[order]) for y in Y[order]])
        return 1 - 3 * sum(abs(r[1:] - r[: n - 1])) / (n**2 - 1)


# -------------------------------------------------------------------------


def ecprint(A: ndarray) -> None:
    """Function that prints the augmented matrix.

    Parameters
    ----------
    A : `np.array`
        The augmented matrix.

    Returns
    -------
    `None`
        Prints the matrix to console.
    """

    n = len(A)
    for i in range(0, n):
        line = ""
    for j in range(0, n + 1):
        line += str(format(round(A[i][j], 2))) + "\t"
        if j == n - 1:
            line += "| "
        print(line)
    print()


def sor_elimination(
    A: ndarray,
    b: ndarray,
    n: int,
    max_iterations: int,
    w: float,
    error: float = 1e-3,
    verbose: bool = True,
) -> ndarray:
    """Computes the Successive Over-Relaxation algorithm.

    Parameters
    ----------
    A : `np.array`
        Coefficient matrix of the system of equations.
    b : `np.array`
        Right-hand side vector of the system of equations.
    n : `int`
        Dimension of the system of equations.
    max_iterations : `int`
        Maximum number of iterations allowed.
    w : `float`
        Relaxation parameter.
    error : `float`, optional
        Desired level of accuracy, default is 1e-3.
    verbose : `bool`, optional
        Whether to print intermediate results, default is False.

    Returns
    -------
    xi : `np.array`
        The solution of the system of equations.
    """
    xin = np.zeros(n)
    for k in range(max_iterations):
        xi = np.zeros(n)
        for i in range(n):
            s1 = np.dot(A[i, :i], xin[:i])
            s2 = np.dot(A[i, i + 1 :], xin[i + 1 :])
            xi[i] = (w / A[i, i]) * (b[i] - s1 - s2) + (1.0 - w) * xin[i]

        difference = np.max(np.abs(xi - xin))
        if verbose:
            print(f"Iteration {k + 1}: xi = {xi}, error = {difference}")
        if difference <= error:
            if verbose:
                print(f"Converged after {k + 1} iterations.")
            return xi
        xin = np.copy(xi)

    raise RuntimeError("Convergence not achieved within the maximum number of iterations.")


def gauss_elimination(A: ndarray | list, pr: int = 2) -> ndarray:
    """Computes the Gauss elimination algorithm.

    Parameters
    ----------
    A : `np.array` or `list`
        An array containing the parameters of the $n$ equations
        with the equalities.

    pr : `int`
        significant numbers of decimals.

    Returns
    -------
    X : `np.array`
        The solution of the system of $n$ equations

    """

    n = len(A)
    X = [0 for _ in range(n)]

    for i in range(n - 1):
        for p in range(i, n):
            if i <= p <= (n - 1) and A[p][i] != 0:
                if p != i:
                    A[p], A[i] = A[i], A[p]
                break
            elif p == (n - 1):
                print("There is no single solution")
                return None

        for j in range(i + 1, n):
            if i <= j <= n and A[j][i] != 0:
                if A[i][i] < A[j][i]:
                    A[j], A[i] = A[i], A[j]
                break

        for j in range(i + 1, n):
            if A[i][i] == 0:
                print("There is no single solution")
                return None
            factor = A[j][i] / A[i][i]
            A[j] = [A[j][k] - factor * A[i][k] for k in range(n + 1)]

    if A[n - 1][n - 1] == 0:
        print("There is no single solution")
        return None

    X[n - 1] = A[n - 1][n] / A[n - 1][n - 1]
    for i in range(n - 2, -1, -1):
        s = sum(A[i][j] * X[j] for j in range(i + 1, n))
        X[i] = (A[i][n] - s) / A[i][i]

    ecprint(A)
    print("The solution is:")
    for i in range(n):
        print(f"\tX{i} = {round(X[i], pr)}")

    return X


# Example usage:
if __name__ == "__main__":
    import pandas as pd

    # Create a sample dataframe with some random
    data = {"x": [3, 5, 7, 9], "y": [4, 6, 8, 2], "z": [1, 2, 1, 3]}
    df = pd.DataFrame(data)
    print("Using the SOR relaxation method : ")
    # Define the coefficient matrix A and the number of variables b
    A = np.array([[1, 1, 1], [1, -1, 2], [1, -1, -3]])
    Ag = A.copy()
    b = np.array([6, 5, -10])
    print("b : ", b)
    # Solve Ax=b, x = [1, 2, 3]
    x = solve(A, b)
    x_hat_sor = sor_elimination(A, b, 3, 200, 0.05)
    # assert np.allclose(x, x_hat_sor), f"Expected:\n{x}\ngot\n{x_hat_sor}"

    print("Using Gaussian elimination : ")
    Ag = np.insert(Ag, len(Ag), b, axis=1)
    print(Ag)
    x_hat_gaus = gauss_elimination(Ag)

    print("New correlation coefficient test")
    X = np.random.rand(100, 1)
    Y = X * X
    print("coefficient for Y = X * X : ", xicor(X, Y))

    print("New correlation coefficient test for pandas DataFrame")
    values_df = xi_corr(df)
    breakpoint()
