import pickle
import warnings
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from likelihood.models import SimulationEngine
from likelihood.tools.numeric_tools import find_multiples

warnings.simplefilter(action="ignore", category=FutureWarning)


class SimpleImputer:
    """Multiple imputation using simulation engine."""

    def __init__(self, n_features: int | None = None, use_scaler: bool = False):
        """
        Initialize the imputer.

        Parameters
        ----------
        n_features: int | None
            Number of features to be used in the imputer. Default is None.
        use_scaler: bool
            Whether to use a scaler. Default is False.
        """
        self.n_features = n_features
        self.sim = SimulationEngine(use_scaler=use_scaler)
        self.params = {}
        self.cols_transf = pd.Series([])

    def fit(self, X: pd.DataFrame) -> None:
        """
        Fit the imputer to the data.

        Parameters
        ----------
        X: pd.DataFrame
            Dataframe to fit the imputer to.
        """
        X_impute = X.copy()
        self.params = self._get_dict_params(X_impute)
        X_impute = self.sim._clean_data(X_impute)

        if X_impute.empty:
            raise ValueError(
                "The dataframe is empty after cleaning, it is not possible to train the imputer."
            )
        self.n_features = self.n_features or X_impute.shape[1] - 1
        self.sim.fit(X_impute, self.n_features)

    def transform(
        self, X: pd.DataFrame, boundary: bool = True, inplace: bool = True
    ) -> pd.DataFrame:
        """
        Impute missing values in the data.

        Parameters
        -----------
        X: pd.DataFrame
            Dataframe to impute missing values.
        boundary: bool
            Whether to use the boundaries of the data to impute missing values. Default is True.
        inplace: bool
            Whether to modify the columns of the original dataframe or return new ones. Default is True.
        """
        X_impute = X.copy()
        self.cols_transf = X_impute.columns
        for column in X_impute.columns:
            if X_impute[column].isnull().sum() > 0:
                if not X_impute[column].dtype == "object":
                    min_value = self.params[column]["min"]
                    max_value = self.params[column]["max"]
                    to_compare = self.params[column]["to_compare"]
                for row in X_impute.index:
                    if pd.isnull(X_impute.loc[row, column]):
                        value_impute = self._check_dtype_convert(
                            self.sim.predict(
                                self._set_zero(X_impute.loc[row, :], column),
                                column,
                            )[0],
                            to_compare,
                        )
                        if not X_impute[column].dtype == "object" and boundary:
                            if value_impute < min_value:
                                value_impute = min_value
                            if value_impute > max_value:
                                value_impute = max_value
                        X_impute.loc[row, column] = value_impute
            else:
                self.cols_transf = self.cols_transf.drop(column)
        if not inplace:
            X_impute = X_impute[self.cols_transf].copy()
            X_impute = X_impute.rename(
                columns={column: column + "_imputed" for column in self.cols_transf}
            )
            X_impute = X.join(X_impute, rsuffix="_imputed")
            order_cols = []
            for column in X.columns:
                if column + "_imputed" in X_impute.columns:
                    order_cols.append(column)
                    order_cols.append(column + "_imputed")
                else:
                    order_cols.append(column)
            X_impute = X_impute[order_cols]
        return X_impute

    def fit_transform(
        self, X: pd.DataFrame, boundary: bool = True, inplace: bool = True
    ) -> pd.DataFrame:
        """
        Fit and transform the data.

        Parameters
        -----------
        X: pd.DataFrame
            Dataframe to fit and transform.
        boundary: bool
            Whether to use the boundaries of the data to impute missing values. Default is True.
        inplace: bool
            Whether to modify the columns of the original dataframe or return new ones. Default is True.
        """
        X_train = X.copy()
        self.fit(X_train)
        return self.transform(X, boundary, inplace)

    def _set_zero(self, X: pd.Series, column_exception) -> pd.DataFrame:
        """
        Set missing values to zero, except for `column_exception`.

        Parameters
        -----------
        X: pd.Series
            Series to set missing values to zero.
        """
        X = X.copy()
        for column in X.index:
            if pd.isnull(X[column]) and column != column_exception:
                X[column] = 0
        data = X.to_frame().T
        return data

    def _check_dtype_convert(self, value: Union[int, float], to_compare: Union[int, float]) -> None:
        """
        Check if the value is an integer and convert it to float if it is.

        Parameters
        -----------
        value: Union[int, float]
            Value to check and convert.
        to_compare: Union[int, float]
            Value to compare to.
        """
        if isinstance(to_compare, int) and isinstance(value, float):
            value = int(round(value, 0))

        if isinstance(to_compare, float) and isinstance(value, float):
            value = round(value, len(str(to_compare).split(".")[1]))
        return value

    def _get_dict_params(self, df: pd.DataFrame) -> dict:
        """
        Get the parameters for the imputer.

        Parameters
        -----------
        df: pd.DataFrame
            Dataframe to get the parameters from.
        """
        params = {}
        for column in df.columns:
            if df[column].isnull().sum() > 0:
                if not df[column].dtype == "object":
                    to_compare = df[column].dropna().sample().values[0]
                    params[column] = {
                        "min": df[column].min(),
                        "to_compare": to_compare,
                        "max": df[column].max(),
                    }
        return params

    def eval(self, X: pd.DataFrame) -> None:
        """
        Create a histogram of the imputed values.

        Parameters
        -----------
        X: pd.DataFrame
            Dataframe to create the histogram from.
        """

        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame.")

        df = X.copy()

        imputed_cols = [col for col in df.columns if col.endswith("_imputed")]
        num_impute = len(imputed_cols)

        if num_impute == 0:
            print("No imputed columns found in the DataFrame.")
            return

        try:
            ncols, nrows = find_multiples(num_impute)
        except ValueError as e:
            print(f"Error finding multiples for {num_impute}: {e}")
            ncols = 1
            nrows = num_impute

        _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 5 * nrows))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for i, col in enumerate(imputed_cols):
            original_col = col.replace("_imputed", "")

            if original_col in df.columns:
                original_col_data = df[original_col].dropna()
                ax = axes[i]

                # Plot the original data
                sns.histplot(
                    original_col_data,
                    kde=True,
                    color="blue",
                    label=f"Original",
                    bins=10,
                    ax=ax,
                )

                # Plot the imputed data
                sns.histplot(
                    df[col],
                    kde=True,
                    color="red",
                    label=f"Imputed",
                    bins=10,
                    ax=ax,
                )

                ax.set_xlabel(original_col)
                ax.set_ylabel("Frequency" if i % ncols == 0 else "")
                ax.legend(loc="upper right")

        plt.suptitle("Histogram Comparison", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()

    def save(self, filename: str = "./imputer") -> None:
        """
        Save the state of the SimpleImputer to a file.

        Parameters
        -----------
        filename: str
            Name of the file to save the imputer to. Default is "./imputer".
        """
        filename = filename if filename.endswith(".pkl") else filename + ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename: str = "./imputer"):
        """
        Load the state of a SimpleImputer from a file.

        Parameters
        -----------
        filename: str
            Name of the file to load the imputer from. Default is "./imputer".
        """
        filename = filename + ".pkl" if not filename.endswith(".pkl") else filename
        with open(filename, "rb") as f:
            return pickle.load(f)
