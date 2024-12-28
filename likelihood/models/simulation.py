import pickle
import warnings
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

from likelihood.tools import DataScaler, FeatureSelection, OneHotEncoder, cdf, check_nan_inf

# Suppress RankWarning
warnings.simplefilter("ignore", np.RankWarning)


# --------------------------------------------------------------------------------------------------------------------------------------
def categories_by_quartile(df: DataFrame, column: str) -> Tuple[str, str]:
    # Count the frequency of each category in the column
    freq = df[column].value_counts()

    # Calculate the 25th percentile (Q1) and 75th percentile (Q3)
    q1 = freq.quantile(0.25)
    q3 = freq.quantile(0.75)

    # Filter categories that are below the 25th percentile and above the 75th percentile
    least_frequent = freq[freq <= q1]
    most_frequent = freq[freq >= q3]

    # Get the least frequent category (25th percentile) and the most frequent category (75th percentile)
    least_frequent_category = least_frequent.idxmin() if not least_frequent.empty else None
    most_frequent_category = most_frequent.idxmax() if not most_frequent.empty else None

    return least_frequent_category, most_frequent_category


class SimulationEngine(FeatureSelection):
    """
    This class implements a predictive model that utilizes multiple linear regression for numerical target variables
    and multiple logistic regression for categorical target variables.

    The class provides methods for training the model on a given dataset, making predictions,
    and evaluating the model's performance.

    Key features:
    - Supports both numerical and categorical target variables, automatically selecting the appropriate regression method.
    - Includes methods for data preprocessing, model fitting, prediction, and evaluation metrics.
    - Designed to be flexible and user-friendly, allowing for easy integration with various datasets.

    Usage:
    - Instantiate the class with the training data and target variable.
    - Call the fit method to train the model.
    - Use the predict method to generate predictions on new data.
    - Evaluate the model using built-in metrics for accuracy and error.

    This class is suitable for applications in data analysis and machine learning, enabling users to leverage regression techniques
    for both numerical and categorical outcomes efficiently.
    """

    def __init__(self, use_scaler: bool = False, **kwargs):

        self.df = pd.DataFrame()
        self.n_importances = None
        self.use_scaler = use_scaler
        self.proba_dict = {}

        super().__init__(**kwargs)

    def predict(self, df: DataFrame, column: str) -> np.ndarray | list:
        # Let us assign the dictionary entries corresponding to the column
        w, quick_encoder, names_cols, dfe, numeric_dict = self.w_dict[column]

        df = df[names_cols].copy()
        # Change the scale of the DataFrame
        dataset = self.df.copy()
        dataset.drop(columns=column, inplace=True)
        numeric_df = dataset.select_dtypes(include="number")
        if self.use_scaler:
            scaler = DataScaler(numeric_df.copy().to_numpy().T, n=None)
            _ = scaler.rescale()
            dataset_ = df.copy()
            numeric_df = dataset_.select_dtypes(include="number")
            numeric_scaled = scaler.rescale(dataset_=numeric_df.to_numpy())
            numeric_df = pd.DataFrame(numeric_scaled.T, columns=numeric_df.columns)
            for col in numeric_df.columns:
                df[col] = numeric_df[col].values

        # Encoding the DataFrame
        for num, colname in enumerate(dfe._encode_columns):
            if df[colname].dtype == "object":
                encode_dict = dfe.encoding_list[num]
                df[colname] = df[colname].apply(
                    dfe._code_transformation_to, dictionary_list=encode_dict
                )

        # Prediction
        y = df.to_numpy() @ w

        # Categorical column
        if quick_encoder != None:

            one_hot = OneHotEncoder()
            y = one_hot.decode(y)
            encoding_dic = quick_encoder.decoding_list[0]
            y = [encoding_dic[item] for item in y]
        # Numeric column
        else:
            if self.use_scaler:
                # scale output
                y += 1
                y /= 2
                y = y * (self.df[column].max() - self.df[column].min())

        return y[:]

    def _encode(self, df: DataFrame) -> np.ndarray | list:
        df = df.copy()
        column = df.columns[0]
        frec = df[column].value_counts() / len(df)
        df.loc[:, "frec"] = df[column].map(frec)
        df.sort_values("frec", inplace=True)
        keys = df[column].to_list()
        values = df["frec"].to_list()
        return dict(zip(keys, values))

    def fit(self, df: DataFrame, n_importances: int, **kwargs) -> None:
        self.df = df
        self.n_importances = n_importances
        # We run the feature selection algorithm
        self.get_digraph(self.df, self.n_importances, self.use_scaler)
        proba_dict_keys = list(self.w_dict.keys())
        self.proba_dict = dict(zip(proba_dict_keys, [i for i in range(len(proba_dict_keys))]))
        for key in proba_dict_keys:
            x = (
                self.df[key].values,
                None if self.df[key].dtype != "object" else self._encode(self.df[[key]]),
            )
            poly = kwargs.get("poly", 9)
            plot = kwargs.get("plot", False)
            if not x[1]:
                media = self.df[key].mean()
                desviacion_estandar = self.df[key].std()
                cota_inferior = media - 1.5 * desviacion_estandar
                cota_superior = media + 1.5 * desviacion_estandar
                if plot:
                    print(f"Cumulative Distribution Function ({key})")
                f, cdf_, ox = cdf(x[0].flatten(), poly=poly, plot=plot)
            else:
                f, ox = None, None
                least_frequent_category, most_frequent_category = categories_by_quartile(
                    self.df[[key]], key
                )
                cota_inferior = x[1].get(least_frequent_category, 0)
                cota_superior = x[1].get(most_frequent_category, 0)
            self.proba_dict[key] = (
                f if f else None,
                x[1],
                (np.mean(np.abs(np.diff(ox))) / 2.0 if isinstance(ox, np.ndarray) else None),
                f(cota_inferior) if f else cota_inferior,
                f(cota_superior) if f else cota_superior,
            )

    def get_proba(self, value: Union[Union[float, int], str] | list, colname: str) -> List[float]:
        value = (
            value
            if isinstance(value, list)
            else value.flatten().tolist() if isinstance(value, np.ndarray) else [value]
        )
        return [
            (
                self.proba_dict[colname][0](val)
                - self.proba_dict[colname][0](val - self.proba_dict[colname][2])
                if (isinstance(val, float) or isinstance(val, int))
                else self.proba_dict[colname][1].get(val, 0)
            )
            for val in value
        ]

    def pred_outliers(self, value: Union[Union[float, int], str] | list, colname: str) -> List[str]:
        return [
            (
                "inlier"
                if (self.proba_dict[colname][3] < val < self.proba_dict[colname][4])
                else "outlier"
            )
            for val in self.get_proba(value, colname)
        ]

    def _clean_data(self, df: DataFrame) -> DataFrame:

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.replace(" ", np.nan, inplace=True)
        df = check_nan_inf(df)
        df = df.reset_index()
        df = df.drop(columns=["index"])

        return df

    def save(self, filename: str = "./simulation_model") -> None:
        """
        Save the state of the SimulationEngine to a file.

        Parameters:
            filename (str): The name of the file where the object will be saved.
        """
        filename = filename if filename.endswith(".pkl") else filename + ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename: str = "./simulation_model"):
        """
        Load the state of a SimulationEngine from a file.

        Parameters:
            filename (str): The name of the file containing the saved object.

        Returns:
            SimulationEngine: A new instance of SimulationEngine with the loaded state.
        """
        filename = filename + ".pkl" if not filename.endswith(".pkl") else filename
        with open(filename, "rb") as f:
            return pickle.load(f)
