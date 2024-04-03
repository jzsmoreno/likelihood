import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from pandas.core.frame import DataFrame

from likelihood.tools import FeatureSelection, OneHotEncoder

# --------------------------------------------------------------------------------------------------------------------------------------


class SimulationEngine(FeatureSelection):

    def __init__(self, df: DataFrame, n_importances: int, **kwargs):

        self.df = df
        self.n_importances = n_importances

        super().__init__(**kwargs)

    def predict(self, column: str, n: int = None) -> ndarray | list:

        # We assing the entries of the dictionary corresponding to the column
        w, quick_encoder, names_cols, dfe = self.w_dict[column]

        df_aux = dfe._df

        if n != None:
            df_aux = df_aux.iloc[:n, :]

        y = df_aux.to_numpy() @ w

        if quick_encoder != None:

            one_hot = OneHotEncoder()
            y = one_hot.decode(y)
            encoding_dic = quick_encoder.decoding_list[0]
            y = [encoding_dic[item] for item in y]

        return y

    def fit(self, **kwargs) -> None:

        # We run the feature selection algorithm
        self.get_digraph(self.df, self.n_importances)
