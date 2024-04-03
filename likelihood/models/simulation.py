import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas.core.frame import DataFrame

from likelihood.tools import DataScaler, FeatureSelection, OneHotEncoder, check_nan_inf

# --------------------------------------------------------------------------------------------------------------------------------------


class SimulationEngine(FeatureSelection):

    def __init__(self, df: DataFrame, n_importances: int, **kwargs):

        self.df = df
        self.n_importances = n_importances

        super().__init__(**kwargs)

    def predict(self, df: DataFrame, column: str, n: int = None) -> ndarray | list:

        # We clean the data set
        df = self._clean_data(df)

        # Let us assign the dictionary entries corresponding to the column
        w, quick_encoder, names_cols, dfe, numeric_dict = self.w_dict[column]

        try:
            df = df[names_cols].copy()
            # Change the scale of the dataframe
            numeric_df = df.select_dtypes(include="number")
            scaler = DataScaler(numeric_df.copy().to_numpy().T, n=None)
            numeric_scaled = scaler.rescale()
            numeric_df = pd.DataFrame(numeric_scaled.T, columns=numeric_df.columns)
            df[numeric_df.columns] = numeric_df

            # Encoding the datadrame
            for num, colname in enumerate(dfe._encode_columns):
                if df[colname].dtype == "object":
                    encode_dict = dfe.encoding_list[num]
                    df[colname] = df[colname].apply(
                        dfe._code_transformation_to, dictionary_list=encode_dict
                    )

        except:
            print("The dataframe provided does not have the same columns as in the fit method.")

        # Assign value to n if n is None
        n = n if n != None else len(df)

        # Generation of assertion
        assert n > 0 and n <= len(df), '"n" must be interger or "<= len(df)".'

        # Sample dataframe
        df_aux = df.sample(n)

        # PREDICTION
        y = df_aux.to_numpy() @ w

        # Categorical column
        if quick_encoder != None:

            one_hot = OneHotEncoder()
            y = one_hot.decode(y)
            encoding_dic = quick_encoder.decoding_list[0]
            y = [encoding_dic[item] for item in y]
        # Numeric column
        else:
            # scale output
            i = numeric_dict[column]
            y += 1
            y /= 2
            y = y * self.scaler.values[1][i]

        return y

    def fit(self, **kwargs) -> None:

        # We run the feature selection algorithm
        self.get_digraph(self.df, self.n_importances)

    def _clean_data(self, df: DataFrame) -> DataFrame:

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.replace(" ", np.nan, inplace=True)
        df = check_nan_inf(df)
        df = df.reset_index()
        df = df.drop(columns=["index"])

        return df
