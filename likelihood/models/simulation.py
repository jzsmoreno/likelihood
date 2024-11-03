import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas.core.frame import DataFrame

from likelihood.tools import DataScaler, FeatureSelection, OneHotEncoder, check_nan_inf

# --------------------------------------------------------------------------------------------------------------------------------------


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

    def __init__(self, df: DataFrame, n_importances: int, use_scaler: bool = False, **kwargs):

        self.df = df
        self.n_importances = n_importances
        self.use_scaler = use_scaler

        super().__init__(**kwargs)

    def predict(self, df: DataFrame, column: str) -> ndarray | list:
        # Let us assign the dictionary entries corresponding to the column
        w, quick_encoder, names_cols, dfe, numeric_dict = self.w_dict[column]

        df = df[names_cols].copy()
        # Change the scale of the dataframe
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

        # Encoding the datadrame
        for num, colname in enumerate(dfe._encode_columns):
            if df[colname].dtype == "object":
                encode_dict = dfe.encoding_list[num]
                df[colname] = df[colname].apply(
                    dfe._code_transformation_to, dictionary_list=encode_dict
                )

        # PREDICTION
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

    def fit(self, **kwargs) -> None:

        # We run the feature selection algorithm
        self.get_digraph(self.df, self.n_importances, self.use_scaler)

    def _clean_data(self, df: DataFrame) -> DataFrame:

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.replace(" ", np.nan, inplace=True)
        df = check_nan_inf(df)
        df = df.reset_index()
        df = df.drop(columns=["index"])

        return df
