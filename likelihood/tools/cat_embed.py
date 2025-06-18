import logging
import os
from typing import List

import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow as tf
from pandas.core.frame import DataFrame
from sklearn.preprocessing import LabelEncoder

tf.get_logger().setLevel("ERROR")


class CategoricalEmbedder:
    def __init__(self, embedding_dim=32):
        self.embedding_dim = embedding_dim
        self.label_encoders = {}
        self.embeddings = {}

    def fit(self, df: DataFrame, categorical_cols: List):
        """
        Fit the embeddings on the given data.

        Parameters
        ----------
        df : `DataFrame`
            Pandas DataFrame containing the tabular data.
        categorical_cols : `List`
            List of column names representing categorical features.

        Returns
        -------
        `None`
        """

        for col in categorical_cols:
            if col not in df.columns:
                raise ValueError(f"Column {col} not found in DataFrame")

        for col in categorical_cols:
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])

        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le

            vocab_size = len(le.classes_)
            embedding_matrix = np.random.rand(vocab_size, self.embedding_dim)
            self.embeddings[col] = tf.Variable(embedding_matrix, dtype=tf.float32)

    def transform(self, df: DataFrame, categorical_cols: List[str]):
        """
        Transform the data using the fitted embeddings.

        Parameters
        ----------
        df : `DataFrame`
            Pandas DataFrame containing the tabular data.
        categorical_cols : `List[str]`
            List of column names representing categorical features.

        Returns
        -------
        Transformed Pandas DataFrame with original columns except `categorical_cols` replaced by their embedding representations.
        """

        df_processed = df.copy()

        for col in categorical_cols:
            if col not in self.label_encoders:
                raise ValueError(
                    f"Column {col} has not been fitted. Please call fit() on this column first."
                )

            le = self.label_encoders[col]
            mode_val = le.transform(le.classes_)[0]
            df_processed[col] = df_processed[col].fillna(mode_val)

        for col in categorical_cols:
            indices_tensor = tf.constant(df_processed[col], dtype=tf.int32)
            embedding_layer = tf.nn.embedding_lookup(
                params=self.embeddings[col], ids=indices_tensor
            )
            if len(embedding_layer.shape) == 1:
                embedding_layer = tf.expand_dims(embedding_layer, axis=0)

            for i in range(self.embedding_dim):
                df_processed[f"{col}_embed_{i}"] = embedding_layer[:, i]
            df_processed.drop(columns=[col], inplace=True)

        return df_processed

    def save_embeddings(self, path: str):
        """
        Save the embeddings to a directory.

        Parameters
        ----------
        path : `str`
            Path to the directory where embeddings will be saved.
        """

        os.makedirs(path, exist_ok=True)
        for col, embedding in self.embeddings.items():
            np.save(os.path.join(path, f"{col}_embedding.npy"), embedding.numpy())

    def load_embeddings(self, path: str):
        """
        Load the embeddings from a directory.

        Parameters
        ----------
        path : `str`
            Path to the directory where embeddings are saved.
        """

        for col in self.label_encoders.keys():
            embedding_path = os.path.join(path, f"{col}_embedding.npy")
            if not os.path.exists(embedding_path):
                raise FileNotFoundError(f"Embedding file {embedding_path} not found.")
            embedding_matrix = np.load(embedding_path)
            self.embeddings[col] = tf.Variable(embedding_matrix, dtype=tf.float32)


if __name__ == "__main__":
    data = {
        "color": ["red", "blue", None, "green", "blue"],
        "size": ["S", "M", "XL", "XS", None],
        "price": [10.99, 25.50, 30.00, 8.75, 12.25],
    }
    df = pd.DataFrame(data)

    # Initialize the embedder
    embedder = CategoricalEmbedder(embedding_dim=3)

    # Fit the embeddings on the data
    embedder.fit(df, categorical_cols=["color", "size"])

    # Transform the data using the fitted embeddings
    processed_df = embedder.transform(df, categorical_cols=["color", "size"])

    print("Processed DataFrame:")
    print(processed_df.head())

    # Save the embeddings to disk
    embedder.save_embeddings("./embeddings")

    # Load the embeddings from disk
    new_embedder = CategoricalEmbedder(embedding_dim=3)
    new_embedder.label_encoders = (
        embedder.label_encoders
    )  # Assuming label encodings are consistent across runs
    new_embedder.load_embeddings("./embeddings")

    # Transform the data using the loaded embeddings
    processed_df_loaded = new_embedder.transform(df, categorical_cols=["color", "size"])
    print("\nProcessed DataFrame with Loaded Embeddings:")
    print(processed_df_loaded.head())
