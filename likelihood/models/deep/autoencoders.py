import os
from functools import partial
from typing import List

import keras_tuner
import numpy as np
import pandas as pd
import tensorflow as tf
from pandas.core.frame import DataFrame
from tensorflow.keras.models import Model

from likelihood.tools import OneHotEncoder


class AutoClassifier(Model):
    def __init__(self, input_shape, num_classes, units, activation):
        super(AutoClassifier, self).__init__()
        self.units = units
        self.shape = input_shape

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(units=units, activation=activation),
                tf.keras.layers.Dense(units=int(units / 2), activation=activation),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(units=units, activation=activation),
                tf.keras.layers.Dense(units=input_shape, activation=activation),
            ]
        )

        self.classifier = tf.keras.Sequential(
            [tf.keras.layers.Dense(num_classes, activation="softmax")]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        combined = tf.concat([decoded, encoded], axis=1)
        classifier = self.classifier(combined)
        return classifier


def call_existing_code(units, activation, threshold, optimizer, input_shape=None, num_classes=None):
    model = AutoClassifier(
        input_shape=input_shape, num_classes=num_classes, units=units, activation=activation
    )
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[tf.keras.metrics.F1Score(threshold=threshold)],
    )
    return model


def build_model(hp, input_shape: None | int, num_classes: None | int):
    units = hp.Int("units", min_value=int(input_shape * 0.2), max_value=input_shape, step=2)
    activation = hp.Choice("activation", ["sigmoid", "relu", "tanh", "selu", "softplus"])
    optimizer = hp.Choice("optimizer", ["sgd", "adam", "adadelta"])
    threshold = hp.Float("threshold", min_value=0.1, max_value=0.9, sampling="log")

    model = call_existing_code(
        units=units,
        activation=activation,
        threshold=threshold,
        optimizer=optimizer,
        input_shape=input_shape,
        num_classes=num_classes,
    )
    return model


def setup_model(
    data: DataFrame, target: str, epochs: int, train_size: float = 0.7, seed=None, **kwargs
) -> AutoClassifier:
    """Setup model for training and tuning."""

    max_trials = kwargs["max_trials"] if "max_trials" in kwargs else 10
    directory = kwargs["directory"] if "directory" in kwargs else "./my_dir"
    project_name = kwargs["project_name"] if "project_name" in kwargs else "get_best"
    objective = kwargs["objective"] if "objective" in kwargs else "val_loss"
    verbose = kwargs["verbose"] if "verbose" in kwargs else True

    X = data.drop(columns=target)
    y = data[target]
    # Verify if there are categorical columns in the dataframe
    assert (
        X.select_dtypes(include=["object"]).empty == False
    ), "Categorical variables within the DataFrame must be encoded, this is done by using the DataFrameEncoder from likelihood."
    validation_split = 1.0 - train_size
    # Create my_dir path if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

        # Create a Classifier instance
        y_encoder = OneHotEncoder()
        y = y_encoder.encode(y.to_list())
        X = X.to_numpy()
        y = pd.DataFrame(y, columns=["class_0", "class_1"])
        y = y.to_numpy()

        input_shape = X.shape[1]
        num_classes = y.shape[1]

        build_model = partial(build_model, input_shape=input_shape, num_classes=num_classes)

        # Create the AutoKeras model
        tuner = keras_tuner.RandomSearch(
            hypermodel=build_model,
            objective=objective,
            max_trials=max_trials,
            directory=directory,
            project_name=project_name,
            seed=seed,
        )

        tuner.search(X, y, epochs=epochs, validation_split=validation_split)
        models = tuner.get_best_models(num_models=2)
        best_model = models[0]

        # save model
        best_model.save("./my_dir/best_model.keras")

        if verbose:
            tuner.results_summary()
    else:
        # Load the best model from the directory
        best_model = tf.keras.models.load_model("./my_dir/best_model.keras")

    return best_model
