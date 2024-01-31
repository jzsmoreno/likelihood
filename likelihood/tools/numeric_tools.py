import numpy as np


def subs(a, b):
    """Function that subtracts lists element by element.

    Parameters
    ----------
    a
    b
    """

    for i, val in enumerate(a):
        val = val - b[i]
        a[i] = val
    return a


def ecprint(A):
    """Function that prints the augmented matrix.

    Parameters
    ----------
    A : np.array
        The augmented matrix.

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


def sor_elimination(A, b, n, nmax, w):
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
        Is a parameter of the SOR method

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

        error = np.max(np.abs(xi - xin))
        print(xi)
        print(f"solution error : {error}")
        if error <= 0.00001:
            print(f"iterations : {k}")
            return xi
        else:
            xin = np.copy(xi)
    return "number of iterations exceeded"


def gauss_elimination(A, pr=2):
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
