import json
import pickle
import re
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from likelihood.tools import generate_html_pipeline
from likelihood.tools.impute import SimpleImputer
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
        self.fitted_components: Dict[str, object] = {}
        self.fitted_idx: List[str] = []
        self.columns_bin_sizes: Dict[str, int] | None = None

    def _load_config(self, config_path: str) -> Dict:
        """Load and validate the JSON configuration."""
        with open(config_path, "r") as f:
            config = json.load(f)

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
            Feature importance scores (if compute_feature_importance=`True`).
        """
        y = df[self.target_col].values
        X = df.drop(columns=[self.target_col]).copy()

        initial_info = {
            "shape": X.shape,
            "columns": list(X.columns),
            "dtypes": X.dtypes.apply(lambda x: x.name).to_dict(),
            "missing_values": X.isnull().sum().to_dict(),
        }

        steps_info = []
        for step in self.steps:
            step_name = step["name"]
            params = step.get("params", {})
            uuid_idx = uuid.uuid4()
            step_info = {
                "step_name": step_name,
                "parameters": params,
                "description": self._get_step_description(step_name),
                "id": uuid_idx,
            }
            step_info["input_columns"] = list(X.columns)
            self.fitted_idx.append(uuid_idx)

            X = self._apply_step(step_name, uuid_idx, X, fit=True, **params)

            step_info["output_shape"] = X.shape
            step_info["output_columns"] = list(X.columns)
            step_info["output_dtypes"] = X.dtypes.apply(lambda x: x.name).to_dict()
            categorical_columns = X.select_dtypes(include=["category"]).columns
            unique_categories = {col: X[col].unique().tolist() for col in categorical_columns}
            step_info["unique_categories"] = unique_categories

            steps_info.append(step_info)

        final_info = {
            "shape": X.shape,
            "columns": list(X.columns),
            "dtypes": X.dtypes.apply(lambda x: x.name).to_dict(),
            "missing_values": X.isnull().sum().to_dict(),
        }

        self.documentation = {
            "initial_dataset": initial_info,
            "processing_steps": steps_info,
            "final_dataset": final_info,
        }

        importances = None
        if self.compute_importance:
            numeric_X = X.select_dtypes(include=["float"])
            numeric_columns = numeric_X.columns.tolist()
            model = LinearRegression()
            model.fit(numeric_X.T.values, y)
            importances = model.get_importances()
            df_scores = pd.DataFrame([importances], columns=numeric_columns)
            df_scores_abs = df_scores.abs()
        df_scores_norm = (
            df_scores_abs / df_scores_abs.to_numpy().sum()
            if isinstance(importances, np.ndarray)
            else pd.DataFrame()
        )
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
        for index, (step_name, _) in enumerate(self.fitted_components.items()):
            step_name = re.sub(r"_[a-f0-9\-]{36}", "", step_name)
            X = self._apply_step(step_name, self.fitted_idx[index], X, fit=False)

        return X

    def get_doc(
        self, save_to_file: bool = True, file_name: str = "data_processing_report.html"
    ) -> None:
        """
        Generate an HTML report from `self.documentation` for pipeline documentation.

        Parameters
        ----------
        save_to_file : bool, optional
            Whether to save generated HTML content to a file. Default is True.
        file_name : str, optional
            Filename for output when `save_to_file` is True. Default is "data_processing_report.html".
        """

        generate_html_pipeline(self.documentation, save_to_file=save_to_file, file_name=file_name)

    def _apply_step(
        self, step_name: str, idx: str, X: pd.DataFrame, fit: bool, **params
    ) -> pd.DataFrame:
        """Dispatch to the correct handler for a preprocessing step."""
        handlers = {
            "DataScaler": self._handle_datascaler,
            "DataFrameEncoder": self._handle_dataframeencoder,
            "remove_collinearity": self._handle_remove_collinearity,
            "TransformRange": self._handle_transformrange,
            "OneHotEncoder": self._handle_onehotencoder,
            "SimpleImputer": self._handle_simpleimputer,
        }

        if step_name not in handlers:
            raise ValueError(
                f"Step '{step_name}' not supported. Supported steps: {list(handlers.keys())}"
            )

        return handlers[step_name](X, idx=idx, fit=fit, **params)

    def _get_step_description(self, step_name: str) -> str:
        """Return a description of what each preprocessing step does."""
        descriptions = {
            "DataScaler": "Scales numerical features using normalization",
            "DataFrameEncoder": "Encodes categorical variables and normalizes to numerical features",
            "remove_collinearity": "Removes highly correlated features to reduce multicollinearity",
            "TransformRange": "Bins continuous features into discrete ranges",
            "OneHotEncoder": "Converts categorical variables into binary variables",
            "SimpleImputer": "Handles missing values by imputing with multiple linear regression strategies",
        }

        return descriptions.get(step_name, f"Unknown preprocessing step: {step_name}")

    # ------------------------------ Step Handlers ------------------------------
    def _handle_datascaler(self, X: pd.DataFrame, idx: str, fit: bool, n: int = 1) -> pd.DataFrame:
        """Handle DataScaler (fits on training data, applies to all)."""
        numeric_X = X.select_dtypes(include=["float"])
        numeric_columns = numeric_X.columns.tolist()
        n = None if n == 0 else n
        if fit:
            scaler = DataScaler(numeric_X.values.T, n=n)
            self.fitted_components[f"DataScaler_{idx}"] = scaler
            numeric_X = pd.DataFrame(scaler.rescale().T, columns=numeric_X.columns)
        else:
            scaler = self.fitted_components[f"DataScaler_{idx}"]
            numeric_X = pd.DataFrame(
                scaler.rescale(numeric_X.values.T).T, columns=numeric_X.columns
            )
        for col in numeric_columns:
            X[col] = numeric_X[col]
        return X

    def _handle_dataframeencoder(
        self, X: pd.DataFrame, idx: str, fit: bool, norm_method: str = "mean"
    ) -> pd.DataFrame:
        """Handle DataFrameEncoder (fits encoders/normalizers)."""
        if fit:
            encoder = DataFrameEncoder(X)
            encoded_X = encoder.encode(norm_method=norm_method)
            self.fitted_components[f"DataFrameEncoder_{idx}"] = encoder
            return encoded_X
        else:
            encoder = self.fitted_components[f"DataFrameEncoder_{idx}"]
            encoder._df = X
            return encoder.encode()

    def _handle_remove_collinearity(
        self, X: pd.DataFrame, idx: str, fit: bool, threshold: float = 0.9
    ) -> pd.DataFrame:
        """Handle collinearity removal (fits by selecting columns to drop)."""
        numeric_X = X.select_dtypes(include=["float"])
        numeric_columns = numeric_X.columns.tolist()
        categorical_columns = set(X.columns) - set(numeric_columns)
        if fit:
            cleaned_X = remove_collinearity(numeric_X, threshold=threshold)
            dropped_cols = set(X.columns) - set(cleaned_X.columns) - categorical_columns
            self.fitted_components[f"remove_collinearity_{idx}"] = dropped_cols
            return X.drop(columns=dropped_cols)
        else:
            dropped_cols = self.fitted_components[f"remove_collinearity_{idx}"]
            return X.drop(columns=dropped_cols)

    def _handle_transformrange(
        self, X: pd.DataFrame, idx: str, fit: bool, columns_bin_sizes: Dict[str, int] | None = None
    ) -> pd.DataFrame:
        """Handle TransformRange (bin numerical features into ranges)."""
        if fit:
            transformer = TransformRange(columns_bin_sizes)
            cleaned_X = transformer.transform(X)
            self.fitted_components[f"TransformRange_{idx}"] = transformer
            self.columns_bin_sizes = columns_bin_sizes
            return cleaned_X
        else:
            transformer = self.fitted_components[f"TransformRange_{idx}"]
            return transformer.transform(X, fit=False)

    def _handle_onehotencoder(
        self, X: pd.DataFrame, idx: str, fit: bool, columns: List[str] | None = None
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
                )
                tmp_df = pd.concat([tmp_df, pd.DataFrame(encoded_X, columns=unique_values)], axis=1)
            self.fitted_components[f"OneHotEncoder_{idx}"] = (encoder, columns, category_to_indices)
        else:
            encoder, columns, category_to_indices = self.fitted_components[f"OneHotEncoder_{idx}"]
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
                )
                tmp_df = pd.concat([tmp_df, pd.DataFrame(encoded_X, columns=unique_values)], axis=1)
                tmp_df[unique_values] = tmp_df[unique_values].fillna(0)
        return tmp_df

    def _handle_simpleimputer(
        self,
        X: pd.DataFrame,
        idx: str,
        fit: bool,
        use_scaler: bool = False,
        boundary: bool = True,
    ) -> pd.DataFrame:
        "Handle SimpleImputer (fit on numerical and categorical columns)."
        if fit:
            use_scaler = True if use_scaler == 1 else False
            imputer = SimpleImputer(use_scaler=use_scaler)
            tmp_df = imputer.fit_transform(X, boundary=boundary)
            self.fitted_components[f"SimpleImputer_{idx}"] = imputer
            return tmp_df
        else:
            imputer = self.fitted_components[f"SimpleImputer_{idx}"]
            return imputer.transform(X, boundary=boundary)

    def save(self, filepath: str) -> None:
        """
        Save the fitted pipeline state to a file using pickle.

        Parameters
        ----------
        filepath : str
            Path where the serialized pipeline will be saved.
        """

        save_dict = {
            "config": self.config,
            "fitted_components": self.fitted_components,
            "fitted_idx": self.fitted_idx,
            "target_col": self.target_col,
            "steps": self.steps,
            "compute_importance": self.compute_importance,
            "columns_bin_sizes": self.columns_bin_sizes,
            "documentation": self.documentation,
        }

        filepath = filepath + ".pkl" if not filepath.endswith(".pkl") else filepath

        with open(filepath, "wb") as f:
            pickle.dump(save_dict, f)

    @classmethod
    def load(cls, filepath: str) -> "Pipeline":
        """
        Load a fitted pipeline from a file.

        Parameters
        ----------
        filepath : str
            Path to the serialized pipeline file.

        Returns
        -------
        pipeline : Pipeline
            Reconstructed pipeline instance with fitted components.
        """

        filepath = filepath + ".pkl" if not filepath.endswith(".pkl") else filepath

        with open(filepath, "rb") as f:
            save_dict = pickle.load(f)

        pipeline = cls.__new__(cls)

        pipeline.config = save_dict["config"]
        pipeline.fitted_components = save_dict["fitted_components"]
        pipeline.fitted_idx = save_dict["fitted_idx"]
        pipeline.target_col = save_dict["target_col"]
        pipeline.steps = save_dict["steps"]
        pipeline.compute_importance = save_dict["compute_importance"]
        pipeline.columns_bin_sizes = save_dict["columns_bin_sizes"]
        pipeline.documentation = save_dict["documentation"]

        return pipeline
