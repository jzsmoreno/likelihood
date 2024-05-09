from numpy import arange, array, ndarray, random
from numpy.linalg import solve
import numpy as np

# -------------------------------------------------------------------------
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


def subs(a: ndarray, b: ndarray) -> ndarray:
    """Function that subtracts lists element by element.

    Parameters
    ----------
    a : `np.array`
        1D Numpy Array.
    b : `np.array`
        1D Numpy Array.

    Returns
    -------
    a : `np.array`
        1D Numpy Array with elements from input arrays subtracted.
    """

    for i, val in enumerate(a):
        val = val - b[i]
        a[i] = val
    return a


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
    A: ndarray, b: ndarray, n: int, nmax: int, w: float, error: float = 1e-9
) -> ndarray:
    """Computes the successive over-relaxation algorithm.

    Parameters
    ----------
    A : `np.array`
        An array containing the parameters of the $n$ equations.
    b : `np.array`
        An array containing the equalities of the $n$ equations.
    n : `int`
        Is the dimension of the system of equations.
    nmax : `int`
        Is the maximum number of iterations.
    w : `float`
        Is a parameter of the SOR method.
    error : `float`
        It is an optional parameter that represents the desired level of accuracy. If not specified, it will be set to `1e-9`.

    Returns
    -------
    xi : `np.array`
        The solution of the system of $n$ equations
    """

    xin = np.zeros(n)
    for k in range(nmax):
        xi = np.zeros(n)
        for i in range(n):
            s1, s2 = 0, 0
            for j in range(i):
                s1 = s1 + (A[i, j] * xin[j])
            for j in range(i + 1, n):
                s2 = s2 + (A[i, j] * xin[j])
            xi[i] = (w / A[i, i]) * (b[i] - s1 - s2) + (1.0 / A[i, i]) * (b[i] - s1) * (1 - w)

        difference = np.max(np.abs(xi - xin))
        print(xi)
        print(f"solution error : {error}")
        if difference <= error:
            print(f"iterations : {k}")
            return xi
        else:
            xin = np.copy(xi)
    return "number of iterations exceeded"


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
    M = [[0 for x in range(n + 1)] for x in range(n)]
    X = [0 for x in range(n)]

    for i in range(n - 1):
        for p in range(i, n):
            if i <= p <= (n - 1) and A[p][i] != 0:
                if p != i:
                    A[p], A[i] = A[i], A[p]
                break
            elif p == (n - 1):
                print("there is no single solution")
                return None

        for j in range(i + 1, n):
            if i <= j <= n and A[j][i] != 0:
                if A[i][i] < A[j][i]:
                    A[j], A[i] = A[i], A[j]
                break

        for j in range(i + 1, n):
            M[j][i] = A[j][i] / A[i][i]
            A[j] = subs(A[j], np.multiply(M[j][i], A[i]))

        if A[n - 1][n - 1] == 0:
            print("there is no single solution")
            return None
        ecprint(A)

        X[n - 1] = A[n - 1][n] / A[n - 1][n - 1]
        for i in list(reversed(range(n - 1))):
            s = 0
            for j in range(i + 1, n):
                s += A[i][j] * X[j]
            X[i] = (A[i][n] - s) / A[i][i]
        print("the solution is:")

        for i in range(n):
            print(f"X{i} = {round(X[i], pr)}")

        return X


# Example usage:
if __name__ == "__main__":
    # Define the coefficient matrix A and the number of variables x
    A = np.array([[3, 2, 7], [4, 6, 5], [1, 8, 9]])
    # Generate a random b
    b = np.random.randint(-100, 100, size=len(A[:, 0]))
    # Solve Ax=b
    x = solve(A, b)
    x_hat = sor_elimination(A, b, 3, 100, 0.1)
    breakpoint()
