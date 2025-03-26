import pickle
from typing import Union

import pandas as pd

from likelihood.models import SimulationEngine


class SimpleImputer:
    """Multiple imputation using simulation engine."""

    def __init__(self, n_features: int | None = None, use_scaler: bool = False):
        """
        Initialize the imputer.

        Parameters
        ----------
        n_features: int | None
            Number of features to impute.
        use_scaler: bool
            Whether to use a scaler.
        """
        self.n_features = n_features
        self.sim = SimulationEngine(use_scaler=use_scaler)

    def fit(self, X: pd.DataFrame) -> None:
        """
        Fit the imputer to the data.

        Parameters
        ----------
        X: pd.DataFrame
            Dataframe to fit the imputer to.
        """
        X_impute = X.copy()
        X_impute = self.sim._clean_data(X_impute)

        if X_impute.empty:
            raise ValueError(
                "The dataframe is empty after cleaning, it is not possible to train the imputer."
            )
        self.n_features = self.n_features or X_impute.shape[1] - 1
        self.sim.fit(X_impute, self.n_features)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values in the data.

        Parameters
        -----------
        X: pd.DataFrame
            Dataframe to impute missing values.
        """
        X_impute = X.copy()
        for column in X_impute.columns:
            if X_impute[column].isnull().sum() > 0:
                to_compare = X_impute[column].dropna().sample().values[0]
                min_value = X_impute[column].min()
                max_value = X_impute[column].max()
                for row in X_impute.index:
                    if pd.isnull(X_impute.loc[row, column]):
                        value_impute = self._check_dtype_convert(
                            self.sim.predict(
                                self._set_zero(X_impute.loc[row, :], column),
                                column,
                            )[0],
                            to_compare,
                        )
                        if value_impute < min_value:
                            value_impute = min_value
                        if value_impute > max_value:
                            value_impute = max_value
                        X_impute.loc[row, column] = value_impute
        return X_impute

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform the data.

        Parameters
        -----------
        X: pd.DataFrame
            Dataframe to fit and transform.
        """
        X_train = X.copy()
        self.fit(X_train)
        return self.transform(X)

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
        if isinstance(to_compare, int) and isinstance(value, int):
            value = float(value)

        if isinstance(to_compare, float) and isinstance(value, float):
            if to_compare.is_integer():
                value = int(value)

        if isinstance(to_compare, float) and isinstance(value, float):
            value = round(value, len(str(to_compare).split(".")[1]))
        return value

    def save(self, filename: str = "./imputer") -> None:
        """
        Save the state of the SimpleImputer to a file.
        """
        filename = filename if filename.endswith(".pkl") else filename + ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename: str = "./imputer"):
        """
        Load the state of a SimpleImputer from a file.
        """
        filename = filename + ".pkl" if not filename.endswith(".pkl") else filename
        with open(filename, "rb") as f:
            return pickle.load(f)
