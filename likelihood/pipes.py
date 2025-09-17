import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from likelihood.tools.models_tools import TransformRange, remove_collinearity
from likelihood.tools.tools import DataFrameEncoder, DataScaler, LinearRegression, OneHotEncoder


class Pipeline:
    def __init__(self, config_path: str):
        """
        Initialize the pipeline with a JSON configuration file.

        Parameters
        ----------
        config_path : str
            Path to the JSON config defining target column and preprocessing steps.
        """
        self.config = self._load_config(config_path)
        self.target_col = self.config["target_column"]
        self.steps = self.config["preprocessing_steps"]
        self.compute_importance = self.config.get("compute_feature_importance", False)
        self.fitted_components: Dict[str, object] = {}  # Stores (step_name, fitted_object)
        self.columns_bin_sizes: Dict[str, int] | None = None

    def _load_config(self, config_path: str) -> Dict:
        """Load and validate the JSON configuration."""
        with open(config_path, "r") as f:
            config = json.load(f)

        # Validate required fields
        assert "target_column" in config, "Config must specify 'target_column'"
        assert "preprocessing_steps" in config, "Config must specify 'preprocessing_steps'"
        return config

    def fit(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, Optional[np.ndarray]]:
        """
        Fit preprocessing components on the input DataFrame and return cleaned X/y.

        Parameters
        ----------
        df : pd.DataFrame
            Input data with features + target column.

        Returns
        -------
        X : pd.DataFrame
            Cleaned feature matrix.
        y : np.ndarray
            Target vector (from self.target_col).
        importances : Optional[np.ndarray]
            Feature importance scores (if compute_feature_importance=True).
        """
        # Extract target and features
        y = df[self.target_col].values
        X = df.drop(columns=[self.target_col]).copy()

        # Apply preprocessing steps
        for step in self.steps:
            step_name = step["name"]
            params = step.get("params", {})
            X = self._apply_step(step_name, X, fit=True, **params)
        # Compute feature importance (if enabled)
        importances = None
        if self.compute_importance:
            numeric_X = X.select_dtypes(include=["float"])
            numeric_columns = numeric_X.columns.tolist()
            model = LinearRegression()
            model.fit(numeric_X.T.values, y)
            importances = model.get_importances()
        df_scores = pd.DataFrame([importances], columns=numeric_columns)
        df_scores_abs = df_scores.abs()
        df_scores_norm = df_scores_abs / df_scores_abs.to_numpy().sum()
        return X, y, df_scores_norm

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted preprocessing steps to new data (no target column needed).

        Parameters
        ----------
        df : pd.DataFrame
            New data to transform.

        Returns
        -------
        X_transformed : pd.DataFrame
            Cleaned feature matrix.
        """
        X = df.copy()
        # Replay fitted steps on new data
        for step_name, _ in self.fitted_components.items():
            X = self._apply_step(step_name, X, fit=False)

        return X

    def _apply_step(self, step_name: str, X: pd.DataFrame, fit: bool, **params) -> pd.DataFrame:
        """Dispatch to the correct handler for a preprocessing step."""
        handlers = {
            "DataScaler": self._handle_datascaler,
            "DataFrameEncoder": self._handle_dataframeencoder,
            "remove_collinearity": self._handle_remove_collinearity,
            "TransformRange": self._handle_transformrange,
            "OneHotEncoder": self._handle_onehotencoder,
        }

        if step_name not in handlers:
            raise ValueError(
                f"Step '{step_name}' not supported. Supported steps: {list(handlers.keys())}"
            )

        return handlers[step_name](X, fit=fit, **params)

    # ------------------------------ Step Handlers ------------------------------
    def _handle_datascaler(self, X: pd.DataFrame, fit: bool, n: int = 1) -> pd.DataFrame:
        """Handle DataScaler (fits on training data, applies to all)."""
        numeric_X = X.select_dtypes(include=["float"])
        numeric_columns = numeric_X.columns.tolist()
        n = None if n == 0 else n
        if fit:
            scaler = DataScaler(numeric_X.values.T, n=n)  # Assume X is numerical for scaling
            self.fitted_components["DataScaler"] = scaler
            numeric_X = pd.DataFrame(scaler.rescale().T, columns=numeric_X.columns)
        else:
            scaler = self.fitted_components["DataScaler"]  # Get latest fitted scaler
            numeric_X = pd.DataFrame(
                scaler.rescale(numeric_X.values.T).T, columns=numeric_X.columns
            )
        for col in numeric_columns:
            X[col] = numeric_X[col]
        return X

    def _handle_dataframeencoder(
        self, X: pd.DataFrame, fit: bool, norm_method: str = "mean"
    ) -> pd.DataFrame:
        """Handle DataFrameEncoder (fits encoders/normalizers)."""
        if fit:
            encoder = DataFrameEncoder(X)
            encoded_X = encoder.encode(norm_method=norm_method)
            self.fitted_components["DataFrameEncoder"] = encoder
            return encoded_X
        else:
            encoder = self.fitted_components["DataFrameEncoder"]
            encoder._df = X
            return encoder.encode()  # Adjust if decode isn't needed—depends on your use case

    def _handle_remove_collinearity(
        self, X: pd.DataFrame, fit: bool, threshold: float = 0.9
    ) -> pd.DataFrame:
        """Handle collinearity removal (fits by selecting columns to drop)."""
        numeric_X = X.select_dtypes(include=["float"])
        numeric_columns = numeric_X.columns.tolist()
        categorical_columns = set(X.columns) - set(numeric_columns)
        if fit:
            cleaned_X = remove_collinearity(numeric_X, threshold=threshold)
            # Store dropped columns to replicate in transform()
            dropped_cols = set(X.columns) - set(cleaned_X.columns) - categorical_columns
            self.fitted_components["remove_collinearity"] = dropped_cols
            return X.drop(columns=dropped_cols)
        else:
            dropped_cols = self.fitted_components["remove_collinearity"]
            return X.drop(columns=dropped_cols)

    def _handle_transformrange(
        self, X: pd.DataFrame, fit: bool, columns_bin_sizes: Dict[str, int] | None = None
    ) -> pd.DataFrame:
        """Handle TransformRange (bin numerical features into ranges)."""
        if fit:
            transformer = TransformRange(X)
            cleaned_X = transformer.transform_dataframe(columns_bin_sizes=columns_bin_sizes)
            self.fitted_components["TransformRange"] = transformer
            self.columns_bin_sizes = columns_bin_sizes
            return cleaned_X
        else:
            transformer = self.fitted_components["TransformRange"]
            transformer.df = X
            return transformer.transform_dataframe(
                columns_bin_sizes=self.columns_bin_sizes, fit=False
            )

    def _handle_onehotencoder(
        self, X: pd.DataFrame, fit: bool, columns: List[str] | None = None
    ) -> pd.DataFrame:
        """Handle OneHotEncoder (fits on categorical columns)."""
        if fit:
            tmp_df = X.drop(columns=columns)
            encoder = OneHotEncoder()
            category_to_indices = {}
            for col in columns:
                unique_values = X[col].unique()
                category_to_indices[col] = {value: i for i, value in enumerate(unique_values)}
                encoded_X = encoder.encode(
                    X[col].values
                    if isinstance(unique_values[0], int)
                    else X[col].map(category_to_indices[col])
                )  # Adjust for multi-column support
                tmp_df = pd.concat([tmp_df, pd.DataFrame(encoded_X, columns=unique_values)], axis=1)
            self.fitted_components["OneHotEncoder"] = (encoder, columns, category_to_indices)
        else:
            encoder, columns, category_to_indices = self.fitted_components["OneHotEncoder"]
            tmp_df = X.drop(columns=columns)
            for col in columns:
                unique_values = list(category_to_indices[col].keys())
                encoded_X = encoder.encode(
                    (
                        X[col].values
                        if isinstance(unique_values[0], int)
                        else X[col].map(category_to_indices[col])
                    ),
                    fit=False,
                )  # Adjust for multi-column support
                tmp_df = pd.concat([tmp_df, pd.DataFrame(encoded_X, columns=unique_values)], axis=1)
        return tmp_df
