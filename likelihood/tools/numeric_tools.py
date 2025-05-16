from typing import Dict

import numpy as np
import pandas as pd
from numpy import arange, array, ndarray, random
from numpy.linalg import solve
from pandas.core.frame import DataFrame


# -------------------------------------------------------------------------
def get_metrics(dataset, actual_column_name, predicted_column_name, verbose=False):
    # Variables to keep track of the number of correct and total predictions
    true_positives = 0  # Correctly predicted positives
    true_negatives = 0  # Correctly predicted negatives
    false_positives = 0  # Negatives predicted as positives
    false_negatives = 0  # Positives predicted as negatives
    total_predictions = len(dataset)

    # Counters for actual and predicted classes
    actual_positive_count = 0
    actual_negative_count = 0
    predicted_positive_count = 0
    predicted_negative_count = 0

    for index, row in dataset.iterrows():
        actual_class = row[actual_column_name]
        predicted_class = row[predicted_column_name]

        # Update confusion matrix counts
        if actual_class == 1 and predicted_class == 1:  # True positive
            true_positives += 1
        elif actual_class == 0 and predicted_class == 0:  # True negative
            true_negatives += 1
        elif actual_class == 0 and predicted_class == 1:  # False positive
            false_positives += 1
        elif actual_class == 1 and predicted_class == 0:  # False negative
            false_negatives += 1

        # Update class counts
        if actual_class == 1:
            actual_positive_count += 1
        else:
            actual_negative_count += 1

        if predicted_class == 1:
            predicted_positive_count += 1
        else:
            predicted_negative_count += 1

    # Calculate accuracy
    accuracy = (true_positives + true_negatives) / total_predictions * 100

    # Calculate precision
    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives) * 100
    else:
        precision = 0  # Avoid division by zero

    # Calculate recall
    if true_positives + false_negatives > 0:
        recall = true_positives / (true_positives + false_negatives) * 100
    else:
        recall = 0  # Avoid division by zero

    # Calculate F1-Score
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0  # Avoid division by zero

    coeff_1 = (true_positives + false_positives) * (false_positives + true_negatives)
    coeff_2 = (true_positives + false_negatives) * (false_negatives + true_negatives)
    if coeff_1 + coeff_2 > 0:
        kappa = (
            2
            * (true_positives * true_negatives - false_negatives * false_positives)
            / (coeff_1 + coeff_2)
        )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "kappa": kappa,
    }

    if verbose:
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Precision: {precision:.2f}%")
        print(f"Recall: {recall:.2f}%")
        print(f"F1-Score: {f1_score:.2f}")
        print(f"Cohen's Kappa: {kappa:.4f}")
    else:
        return metrics


def xi_corr(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate new coefficient of correlation for all pairs of columns in a `DataFrame`.

    Parameters
    ----------
    df : `DataFrame`
        Input data containing the variables to be correlated.

    Returns
    -------
    `DataFrame`
        A square dataframe with variable names as both index and columns,
        containing their corresponding correlation coefficients.
    """

    columns = df.select_dtypes(include="number").columns
    n = len(columns)

    # Initialize a square matrix for the correlations
    correlations = pd.DataFrame(1.0, index=columns, columns=columns)

    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            if i < j:
                x = df[col1].values
                y = df[col2].values

                correlation = xicor(x, y)
                correlations.loc[col1, col2] = round(correlation, 8)
                correlations.loc[col2, col1] = round(correlation, 8)  # Mirror the correlation

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


def xicor(X: np.ndarray, Y: np.ndarray, ties: bool = True, random_seed: int = None) -> float:
    """
    Calculate a generalized coefficient of correlation between two variables.

    This coefficient is an extension of Pearson's correlation, accounting for ties with optional randomization.

    Parameters
    ----------
    X : `np.ndarray`
        The first variable to be correlated. Must have at least one dimension.
    Y : `np.ndarray`
        The second variable to be correlated. Must have at least one dimension.
    ties : bool
        Whether to handle ties using randomization.
    random_seed : int, optional
        Seed for the random number generator for reproducibility.

    Returns
    -------
    xi : `float`
        The estimated value of the new coefficient of correlation.
    """

    # Early return for identical arrays
    if np.array_equal(X, Y):
        return 1.0

    n = len(X)

    # Early return for cases with less than 2 elements
    if n < 2:
        return 0.0

    # Flatten the input arrays if they are multidimensional
    X = X.flatten()
    Y = Y.flatten()

    # Get the sorted order of X
    order = np.argsort(X)

    if ties:
        np.random.seed(random_seed)  # Set seed for reproducibility if needed
        ranks = np.argsort(np.argsort(Y[order]))  # Get ranks
        unique_ranks, counts = np.unique(ranks, return_counts=True)

        # Adjust ranks for ties by shuffling
        for rank, count in zip(unique_ranks, counts):
            if count > 1:
                tie_indices = np.where(ranks == rank)[0]
                np.random.shuffle(ranks[tie_indices])  # Randomize ties

        cumulative_counts = np.array([np.sum(y >= Y[order]) for y in Y[order]])
        return 1 - n * np.sum(np.abs(ranks[1:] - ranks[: n - 1])) / (
            2 * np.sum(cumulative_counts * (n - cumulative_counts))
        )
    else:
        ranks = np.argsort(np.argsort(Y[order]))  # Get ranks without randomization
        return 1 - 3 * np.sum(np.abs(ranks[1:] - ranks[: n - 1])) / (n**2 - 1)


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


def find_multiples(target: int) -> tuple[int, int] | None:
    """Find two factors of a given target number.

    Parameters
    ----------
    target : int
        The target number to find factors for.

    Returns
    -------
    tuple[int, int] | None
        If i and i+1 both divide target, returns (i, i+1).
        Otherwise, returns (i, target // i).
        Returns None if no factors are found.
    """
    for i in range(2, target + 1):
        if target % i == 0:
            if (i + 1) <= target and target % (i + 1) == 0:
                return i + 1, target // (i + 1)
            else:
                return i, target // i
    return None


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
    print("coefficient for Y = X * X : ", xicor(X, Y, False))
    df["index"] = ["A", "B", "C", "D"]
    print("New correlation coefficient test for pandas DataFrame")
    values_df = xi_corr(df)
    print(find_multiples(30))
    print(find_multiples(25))
    print(find_multiples(49))
    print(find_multiples(17))
    print(find_multiples(24))
    breakpoint()
