import logging
import os
from functools import partial
from shutil import rmtree

import keras_tuner
import numpy as np
import pandas as pd
import tensorflow as tf
from pandas.core.frame import DataFrame

from likelihood.tools import OneHotEncoder

logging.getLogger("tensorflow").setLevel(logging.ERROR)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


@tf.keras.utils.register_keras_serializable(package="Custom", name="AutoClassifier")
class AutoClassifier(tf.keras.Model):
    """
    An auto-classifier model that automatically determines the best classification strategy based on the input data.

    Attributes:
        - input_shape_parm: The shape of the input data.
        - num_classes: The number of classes in the dataset.
        - units: The number of neurons in each hidden layer.
        - activation: The type of activation function to use for the neural network layers.

    Methods:
        __init__(self, input_shape_parm, num_classes, units, activation): Initializes an AutoClassifier instance with the given parameters.
        build(self, input_shape_parm): Builds the model architecture based on input_shape_parm.
        call(self, x): Defines the forward pass of the model.
        get_config(self): Returns the configuration of the model.
        from_config(cls, config): Recreates an instance of AutoClassifier from its configuration.
    """

    def __init__(self, input_shape_parm, num_classes, units, activation):
        """
        Initializes an AutoClassifier instance with the given parameters.

        Parameters
        ----------
        input_shape_parm : `int`
            The shape of the input data.
        num_classes : `int`
            The number of classes in the dataset.
        units : `int`
            The number of neurons in each hidden layer.
        activation : `str`
            The type of activation function to use for the neural network layers.
        """
        super(AutoClassifier, self).__init__()
        self.input_shape_parm = input_shape_parm
        self.num_classes = num_classes
        self.units = units
        self.activation = activation

        self.encoder = None
        self.decoder = None
        self.classifier = None

    def build(self, input_shape):
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(units=self.units, activation=self.activation),
                tf.keras.layers.Dense(units=int(self.units / 2), activation=self.activation),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(units=self.units, activation=self.activation),
                tf.keras.layers.Dense(units=self.input_shape_parm, activation=self.activation),
            ]
        )

        self.classifier = tf.keras.Sequential(
            [tf.keras.layers.Dense(self.num_classes, activation="softmax")]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        combined = tf.concat([decoded, encoded], axis=1)
        classification = self.classifier(combined)
        return classification

    def get_config(self):
        config = {
            "input_shape_parm": self.input_shape_parm,
            "num_classes": self.num_classes,
            "units": self.units,
            "activation": self.activation,
        }
        base_config = super(AutoClassifier, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(
            input_shape_parm=config["input_shape_parm"],
            num_classes=config["num_classes"],
            units=config["units"],
            activation=config["activation"],
        )


def call_existing_code(
    units: int,
    activation: str,
    threshold: float,
    optimizer: str,
    input_shape_parm: None | int = None,
    num_classes: None | int = None,
) -> AutoClassifier:
    """
    Calls an existing AutoClassifier instance.

    Parameters
    ----------
    units : `int`
        The number of neurons in each hidden layer.
    activation : `str`
        The type of activation function to use for the neural network layers.
    threshold : `float`
        The threshold for the classifier.
    optimizer : `str`
        The type of optimizer to use for the neural network layers.
    input_shape_parm : `None` | `int`
        The shape of the input data.
    num_classes : `int`
        The number of classes in the dataset.

    Returns
    -------
    `AutoClassifier`
        The AutoClassifier instance.
    """
    model = AutoClassifier(
        input_shape_parm=input_shape_parm,
        num_classes=num_classes,
        units=units,
        activation=activation,
    )
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.F1Score(threshold=threshold)],
    )
    return model


def build_model(hp, input_shape_parm: None | int, num_classes: None | int) -> AutoClassifier:
    """Builds a neural network model using Keras Tuner's search algorithm.

    Parameters
    ----------
    hp : `keras_tuner.HyperParameters`
        The hyperparameters to tune.
    input_shape_parm : `None` | `int`
        The shape of the input data.
    num_classes : `int`
        The number of classes in the dataset.

    Returns
    -------
    `keras.Model`
        The neural network model.
    """
    units = hp.Int(
        "units", min_value=int(input_shape_parm * 0.2), max_value=input_shape_parm, step=2
    )
    activation = hp.Choice("activation", ["sigmoid", "relu", "tanh", "selu", "softplus"])
    optimizer = hp.Choice("optimizer", ["sgd", "adam", "adadelta"])
    threshold = hp.Float("threshold", min_value=0.1, max_value=0.9, sampling="log")

    model = call_existing_code(
        units=units,
        activation=activation,
        threshold=threshold,
        optimizer=optimizer,
        input_shape_parm=input_shape_parm,
        num_classes=num_classes,
    )
    return model


def setup_model(
    data: DataFrame,
    target: str,
    epochs: int,
    train_size: float = 0.7,
    seed=None,
    train_mode: bool = True,
    filepath: str = "./my_dir/best_model",
    **kwargs,
) -> AutoClassifier:
    """Setup model for training and tuning.

    Parameters
    ----------
    data : `DataFrame`
        The dataset to train the model on.
    target : `str`
        The name of the target column.
    epochs : `int`
        The number of epochs to train the model for.
    train_size : `float`
        The proportion of the dataset to use for training.
    seed : `Any` | `int`
        The random seed to use for reproducibility.
    train_mode : `bool`
        Whether to train the model or not.
    filepath : `str`
        The path to save the best model to.

    Keyword Arguments:
    ----------
    Additional keyword arguments to pass to the model.

    max_trials : `int`
        The maximum number of trials to perform.
    directory : `str`
        The directory to save the model to.
    project_name : `str`
        The name of the project.
    objective : `str`
        The objective to optimize.
    verbose : `bool`
        Whether to print verbose output.

    Returns
    -------
    model : `AutoClassifier`
        The trained model.
    """
    max_trials = kwargs["max_trials"] if "max_trials" in kwargs else 10
    directory = kwargs["directory"] if "directory" in kwargs else "./my_dir"
    project_name = kwargs["project_name"] if "project_name" in kwargs else "get_best"
    objective = kwargs["objective"] if "objective" in kwargs else "val_loss"
    verbose = kwargs["verbose"] if "verbose" in kwargs else True

    X = data.drop(columns=target)
    input_sample = X.sample(1)
    y = data[target]
    # Verify if there are categorical columns in the dataframe
    assert (
        X.select_dtypes(include=["object"]).empty == True
    ), "Categorical variables within the DataFrame must be encoded, this is done by using the DataFrameEncoder from likelihood."
    validation_split = 1.0 - train_size
    # Create my_dir path if it does not exist

    if train_mode:
        # Create a new directory if it does not exist
        try:
            if (not os.path.exists(directory)) and directory != "./":
                os.makedirs(directory)
            elif directory != "./":
                print(f"Directory {directory} already exists, it will be deleted.")
                rmtree(directory)
                os.makedirs(directory)
        except:
            print("Warning: unable to create directory")

        # Create a Classifier instance
        y_encoder = OneHotEncoder()
        y = y_encoder.encode(y.to_list())
        X = X.to_numpy()
        input_sample.to_numpy()
        X = np.asarray(X).astype(np.float32)
        input_sample = np.asarray(input_sample).astype(np.float32)
        y = np.asarray(y).astype(np.float32)

        input_shape_parm = X.shape[1]
        num_classes = y.shape[1]
        global build_model
        build_model = partial(
            build_model, input_shape_parm=input_shape_parm, num_classes=num_classes
        )

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
        best_model(input_sample)

        # save model
        best_model.save(filepath, save_format="tf")

        if verbose:
            tuner.results_summary()
    else:
        # Load the best model from the directory
        best_model = tf.keras.models.load_model(filepath)

    return best_model


########################################################################################
