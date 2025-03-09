import logging
import os
import random
import warnings
from functools import partial
from shutil import rmtree

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import radviz

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)


import keras_tuner
import tensorflow as tf
from pandas.core.frame import DataFrame
from sklearn.manifold import TSNE
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.regularizers import l2

from likelihood.tools import LoRALayer, OneHotEncoder, suppress_warnings

tf.get_logger().setLevel("ERROR")


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0
        self.stop_training = False

    def __call__(self, current_loss):
        if self.best_loss - current_loss > self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.stop_training = True


def mse_loss(y_true, y_pred):
    """
    Mean squared error loss function.

    Parameters
    ----------
    y_true : `tf.Tensor`
        The true values.
    y_pred : `tf.Tensor`
        The predicted values.

    Returns
    -------
    `tf.Tensor`
    """
    return tf.reduce_mean(tf.square(y_true - y_pred))


def kl_loss(mean, log_var):
    """
    Kullback-Leibler divergence loss function.

    Parameters
    ----------
    mean : `tf.Tensor`
        The mean of the distribution.
    log_var : `tf.Tensor`
        The log variance of the distribution.

    Returns
    -------
    `tf.Tensor`
    """
    return -0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var))


def vae_loss(y_true, y_pred, mean, log_var):
    """
    Variational autoencoder loss function.

    Parameters
    ----------
    y_true : `tf.Tensor`
        The true values.
    y_pred : `tf.Tensor`
        The predicted values.
    mean : `tf.Tensor`
        The mean of the distribution.
    log_var : `tf.Tensor`
        The log variance of the distribution.

    Returns
    -------
    `tf.Tensor`
    """
    return mse_loss(y_true, y_pred) + kl_loss(mean, log_var)


def sampling(mean, log_var, epsilon_value=1e-8):
    """
    Samples from the distribution.

    Parameters
    ----------
    mean : `tf.Tensor`
        The mean of the distribution.
    log_var : `tf.Tensor`
        The log variance of the distribution.
    epsilon_value : float
        A small value to avoid numerical instability.

    Returns
    -------
    `tf.Tensor`
    """
    epsilon = tf.random.normal(shape=tf.shape(mean), mean=0.0, stddev=1.0)
    stddev = tf.exp(0.5 * log_var) + epsilon_value
    epsilon = tf.random.normal(shape=tf.shape(mean), mean=0.0, stddev=1.0)
    return mean + stddev * epsilon


def check_for_nans(tensors, name="Tensor"):
    for t in tensors:
        if tf.reduce_any(tf.math.is_nan(t)) or tf.reduce_any(tf.math.is_inf(t)):
            print(f"Warning: {name} contains NaNs or Infs")
            return True
    return False


def cal_loss_step(batch, encoder, decoder, vae_mode=False, training=True):
    """
    Calculates the loss value on a batch of data.

    Parameters
    ----------
    batch : `tf.Tensor`
        The batch of data.
    encoder : `tf.keras.Model`
        The encoder model.
    decoder : `tf.keras.Model`
        The decoder model.
    optimizer : `tf.keras.optimizers.Optimizer`
        The optimizer to use.
    vae_mode : `bool`
        Whether to use variational autoencoder mode. Default is False.
    training : `bool`
        Whether the model is in training mode. Default is True.

    Returns
    -------
    `tf.Tensor`
        The loss value.
    """
    if vae_mode:
        mean, log_var = encoder(batch, training=training)
        log_var = tf.clip_by_value(log_var, clip_value_min=1e-8, clip_value_max=tf.float32.max)
        decoded = decoder(sampling(mean, log_var), training=training)
        loss = vae_loss(batch, decoded, mean, log_var)
    else:
        encoded = encoder(batch, training=training)
        decoded = decoder(encoded, training=training)
        loss = mse_loss(batch, decoded)

    return loss


@tf.function
def train_step(batch, encoder, decoder, optimizer, vae_mode=False):
    """
    Trains the model on a batch of data.

    Parameters
    ----------
    mean : `tf.Tensor`
        The mean of the distribution.
    log_var : `tf.Tensor`
        The log variance of the distribution.
    batch : `tf.Tensor`
        The batch of data.
    encoder : `tf.keras.Model`
        The encoder model.
    decoder : `tf.keras.Model`
        The decoder model.
    optimizer : `tf.keras.optimizers.Optimizer`
        The optimizer to use.
    vae_mode : `bool`
        Whether to use variational autoencoder mode. Default is False.

    Returns
    -------
    `tf.Tensor`
        The loss value.
    """
    optimizer.build(encoder.trainable_variables + decoder.trainable_variables)

    with tf.GradientTape() as encoder_tape, tf.GradientTape() as decoder_tape:
        loss = cal_loss_step(batch, encoder, decoder, vae_mode=vae_mode)

    gradients_of_encoder = encoder_tape.gradient(loss, encoder.trainable_variables)
    gradients_of_decoder = decoder_tape.gradient(loss, decoder.trainable_variables)

    optimizer.apply_gradients(zip(gradients_of_encoder, encoder.trainable_variables))
    optimizer.apply_gradients(zip(gradients_of_decoder, decoder.trainable_variables))

    return loss


@tf.keras.utils.register_keras_serializable(package="Custom", name="AutoClassifier")
class AutoClassifier(tf.keras.Model):
    """
    An auto-classifier model that automatically determines the best classification strategy based on the input data.

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

    Keyword Arguments:
    ----------
    Additional keyword arguments to pass to the model.

    classifier_activation : `str`
        The activation function to use for the classifier layer. Default is "softmax". If the activation function is not a classification function, the model can be used in regression problems.
    num_layers : `int`
        The number of hidden layers in the classifier. Default is 1.
    dropout : `float`
        The dropout rate to use in the classifier. Default is None.
    l2_reg : `float`
        The L2 regularization parameter. Default is 0.0.
    vae_mode : `bool`
        Whether to use variational autoencoder mode. Default is False.
    vae_units : `int`
        The number of units in the variational autoencoder. Default is 2.
    lora_mode : `bool`
        Whether to use LoRA layers. Default is False.
    lora_rank : `int`
        The rank of the LoRA layer. Default is 4.
    """

    def __init__(self, input_shape_parm, num_classes, units, activation, **kwargs):
        super(AutoClassifier, self).__init__()
        self.input_shape_parm = input_shape_parm
        self.num_classes = num_classes
        self.units = units
        self.activation = activation

        self.encoder = None
        self.decoder = None
        self.classifier = None
        self.classifier_activation = kwargs.get("classifier_activation", "softmax")
        self.num_layers = kwargs.get("num_layers", 1)
        self.dropout = kwargs.get("dropout", None)
        self.l2_reg = kwargs.get("l2_reg", 0.0)
        self.vae_mode = kwargs.get("vae_mode", False)
        self.vae_units = kwargs.get("vae_units", 2)
        self.lora_mode = kwargs.get("lora_mode", False)
        self.lora_rank = kwargs.get("lora_rank", 4)

    def build_encoder_decoder(self, input_shape):
        self.encoder = (
            tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(
                        units=self.units,
                        activation=self.activation,
                        kernel_regularizer=l2(self.l2_reg),
                    ),
                    tf.keras.layers.Dense(
                        units=int(self.units / 2),
                        activation=self.activation,
                        kernel_regularizer=l2(self.l2_reg),
                    ),
                ]
            )
            if not self.encoder
            else self.encoder
        )

        self.decoder = (
            tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(
                        units=self.units,
                        activation=self.activation,
                        kernel_regularizer=l2(self.l2_reg),
                    ),
                    tf.keras.layers.Dense(
                        units=self.input_shape_parm,
                        activation=self.activation,
                        kernel_regularizer=l2(self.l2_reg),
                    ),
                ]
            )
            if not self.decoder
            else self.decoder
        )

    def build(self, input_shape):
        if self.vae_mode:
            inputs = tf.keras.Input(shape=self.input_shape_parm, name="encoder_input")
            x = tf.keras.layers.Dense(
                units=self.units,
                kernel_regularizer=l2(self.l2_reg),
                kernel_initializer="he_normal",
            )(inputs)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(self.activation)(x)
            x = tf.keras.layers.Dense(
                units=int(self.units / 2),
                kernel_regularizer=l2(self.l2_reg),
                kernel_initializer="he_normal",
                name="encoder_hidden",
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(self.activation)(x)

            mean = tf.keras.layers.Dense(2, name="mean")(x)
            log_var = tf.keras.layers.Dense(2, name="log_var")(x)
            log_var = tf.keras.layers.Lambda(lambda x: x + 1e-7)(log_var)

            self.encoder = (
                tf.keras.Model(inputs, [mean, log_var], name="encoder")
                if not self.encoder
                else self.encoder
            )
            self.decoder = (
                tf.keras.Sequential(
                    [
                        tf.keras.layers.Dense(
                            units=self.units,
                            kernel_regularizer=l2(self.l2_reg),
                        ),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Activation(self.activation),
                        tf.keras.layers.Dense(
                            units=self.input_shape_parm,
                            kernel_regularizer=l2(self.l2_reg),
                        ),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Activation(self.activation),
                    ]
                )
                if not self.decoder
                else self.decoder
            )

        else:
            self.build_encoder_decoder(input_shape)

        # Classifier with L2 regularization
        self.classifier = tf.keras.Sequential()
        if self.num_layers > 1 and not self.lora_mode:
            for _ in range(self.num_layers - 1):
                self.classifier.add(
                    tf.keras.layers.Dense(
                        units=self.units,
                        activation=self.activation,
                        kernel_regularizer=l2(self.l2_reg),
                    )
                )
                if self.dropout:
                    self.classifier.add(tf.keras.layers.Dropout(self.dropout))
            self.classifier.add(
                tf.keras.layers.Dense(
                    units=self.num_classes,
                    activation=self.classifier_activation,
                    kernel_regularizer=l2(self.l2_reg),
                )
            )
        elif self.lora_mode:
            for _ in range(self.num_layers - 1):
                self.classifier.add(
                    LoRALayer(units=self.units, rank=self.lora_rank, name=f"LoRA_{_}")
                )
                self.classifier.add(tf.keras.layers.Activation(self.activation))
                if self.dropout:
                    self.classifier.add(tf.keras.layers.Dropout(self.dropout))
            self.classifier.add(
                tf.keras.layers.Dense(
                    units=self.num_classes,
                    activation=self.classifier_activation,
                    kernel_regularizer=l2(self.l2_reg),
                )
            )
        else:
            self.classifier.add(
                tf.keras.layers.Dense(
                    units=self.num_classes,
                    activation=self.classifier_activation,
                    kernel_regularizer=l2(self.l2_reg),
                )
            )

    def train_encoder_decoder(
        self, data, epochs, batch_size, validation_split=0.2, patience=10, **kwargs
    ):
        """
        Trains the encoder and decoder on the input data.

        Parameters
        ----------
        data : `tf.data.Dataset`, `np.ndarray`
            The input data.
        epochs : `int`
            The number of epochs to train for.
        batch_size : `int`
            The batch size to use.
        validation_split : `float`
            The proportion of the dataset to use for validation. Default is 0.2.
        patience : `int`
            The number of epochs to wait before early stopping. Default is 10.

        Keyword Arguments:
        ----------
        Additional keyword arguments to pass to the model.
        """
        verbose = kwargs.get("verbose", True)
        optimizer = kwargs.get("optimizer", tf.keras.optimizers.Adam())
        dummy_input = tf.convert_to_tensor(tf.random.normal([1, self.input_shape_parm]))
        self.build(dummy_input.shape)
        if not self.vae_mode:
            dummy_output = self.encoder(dummy_input)
            self.decoder(dummy_output)
        else:
            mean, log_var = self.encoder(dummy_input)
            dummy_output = sampling(mean, log_var)
            self.decoder(dummy_output)

        if isinstance(data, np.ndarray):
            data = tf.data.Dataset.from_tensor_slices(data).batch(batch_size)
            data = data.map(lambda x: tf.cast(x, tf.float32))

        early_stopping = EarlyStopping(patience=patience)
        train_batches = data.take(int((1 - validation_split) * len(data)))
        val_batches = data.skip(int((1 - validation_split) * len(data)))
        for epoch in range(epochs):
            for train_batch, val_batch in zip(train_batches, val_batches):
                loss_train = train_step(
                    train_batch, self.encoder, self.decoder, optimizer, self.vae_mode
                )
                loss_val = cal_loss_step(
                    val_batch, self.encoder, self.decoder, self.vae_mode, False
                )

            early_stopping(loss_train)

            if early_stopping.stop_training:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

            if epoch % 10 == 0 and verbose:
                print(
                    f"Epoch {epoch}: Train Loss: {loss_train:.6f} Validation Loss: {loss_val:.6f}"
                )
        self.freeze_encoder_decoder()

    def call(self, x):
        if self.vae_mode:
            mean, log_var = self.encoder(x)
            encoded = sampling(mean, log_var)
        else:
            encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        combined = tf.concat([decoded, encoded], axis=1)
        classification = self.classifier(combined)
        return classification

    def freeze_encoder_decoder(self):
        """
        Freezes the encoder and decoder layers to prevent them from being updated during training.
        """
        for layer in self.encoder.layers:
            layer.trainable = False
        for layer in self.decoder.layers:
            layer.trainable = False

    def unfreeze_encoder_decoder(self):
        """
        Unfreezes the encoder and decoder layers allowing them to be updated during training.
        """
        for layer in self.encoder.layers:
            layer.trainable = True
        for layer in self.decoder.layers:
            layer.trainable = True

    def set_encoder_decoder(self, source_model):
        """
        Sets the encoder and decoder layers from another AutoClassifier instance,
        ensuring compatibility in dimensions. Only works if vae_mode is False.

        Parameters:
        -----------
        source_model : AutoClassifier
            The source model to copy the encoder and decoder layers from.

        Raises:
        -------
        ValueError
            If the input shape or units of the source model do not match.
        """
        if not isinstance(source_model, AutoClassifier):
            raise ValueError("Source model must be an instance of AutoClassifier.")

        # Check compatibility in input shape and units
        if self.input_shape_parm != source_model.input_shape_parm:
            raise ValueError(
                f"Incompatible input shape. Expected {self.input_shape_parm}, got {source_model.input_shape_parm}."
            )
        if self.units != source_model.units:
            raise ValueError(
                f"Incompatible number of units. Expected {self.units}, got {source_model.units}."
            )
        self.encoder, self.decoder = tf.keras.Sequential(), tf.keras.Sequential()
        # Copy the encoder layers
        for i, layer in enumerate(source_model.encoder.layers):
            if isinstance(layer, tf.keras.layers.Dense):  # Make sure it's a Dense layer
                dummy_input = tf.convert_to_tensor(tf.random.normal([1, layer.input_shape[1]]))
                dense_layer = tf.keras.layers.Dense(
                    units=layer.units,
                    activation=self.activation,
                    kernel_regularizer=l2(self.l2_reg),
                )
                dense_layer.build(dummy_input.shape)
                self.encoder.add(dense_layer)
                # Set the weights correctly
                self.encoder.layers[i].set_weights(layer.get_weights())
            elif not isinstance(layer, InputLayer):
                raise ValueError(f"Layer type {type(layer)} not supported for copying.")

        # Copy the decoder layers
        for i, layer in enumerate(source_model.decoder.layers):
            if isinstance(layer, tf.keras.layers.Dense):  # Ensure it's a Dense layer
                dummy_input = tf.convert_to_tensor(tf.random.normal([1, layer.input_shape[1]]))
                dense_layer = tf.keras.layers.Dense(
                    units=layer.units,
                    activation=self.activation,
                    kernel_regularizer=l2(self.l2_reg),
                )
                dense_layer.build(dummy_input.shape)
                self.decoder.add(dense_layer)
                # Set the weights correctly
                self.decoder.layers[i].set_weights(layer.get_weights())
            elif not isinstance(layer, InputLayer):
                raise ValueError(f"Layer type {type(layer)} not supported for copying.")

    def get_config(self):
        config = {
            "input_shape_parm": self.input_shape_parm,
            "num_classes": self.num_classes,
            "units": self.units,
            "activation": self.activation,
            "classifier_activation": self.classifier_activation,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "l2_reg": self.l2_reg,
            "vae_mode": self.vae_mode,
            "vae_units": self.vae_units,
            "lora_mode": self.lora_mode,
            "lora_rank": self.lora_rank,
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
            classifier_activation=config["classifier_activation"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            l2_reg=config["l2_reg"],
            vae_mode=config["vae_mode"],
            vae_units=config["vae_units"],
            lora_mode=config["lora_mode"],
            lora_rank=config["lora_rank"],
        )


def call_existing_code(
    units: int,
    activation: str,
    threshold: float,
    optimizer: str,
    input_shape_parm: None | int = None,
    num_classes: None | int = None,
    num_layers: int = 1,
    **kwargs,
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
    num_layers : `int`
        The number of hidden layers in the classifier. Default is 1.

    Returns
    -------
    `AutoClassifier`
        The AutoClassifier instance.
    """
    dropout = kwargs.get("dropout", None)
    l2_reg = kwargs.get("l2_reg", 0.0)
    model = AutoClassifier(
        input_shape_parm=input_shape_parm,
        num_classes=num_classes,
        units=units,
        activation=activation,
        num_layers=num_layers,
        dropout=dropout,
        l2_reg=l2_reg,
    )
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.F1Score(threshold=threshold)],
    )
    return model


def build_model(
    hp, input_shape_parm: None | int, num_classes: None | int, **kwargs
) -> AutoClassifier:
    """Builds a neural network model using Keras Tuner's search algorithm.

    Parameters
    ----------
    hp : `keras_tuner.HyperParameters`
        The hyperparameters to tune.
    input_shape_parm : `None` | `int`
        The shape of the input data.
    num_classes : `int`
        The number of classes in the dataset.

    Keyword Arguments:
    ----------
    Additional keyword arguments to pass to the model.

    hyperparameters : `dict`
        The hyperparameters to set.

    Returns
    -------
    `keras.Model`
        The neural network model.
    """
    hyperparameters = kwargs.get("hyperparameters", None)
    hyperparameters_keys = hyperparameters.keys() if hyperparameters is not None else []

    units = (
        hp.Int(
            "units",
            min_value=int(input_shape_parm * 0.2),
            max_value=int(input_shape_parm * 1.5),
            step=2,
        )
        if "units" not in hyperparameters_keys
        else (
            hp.Choice("units", hyperparameters["units"])
            if isinstance(hyperparameters["units"], list)
            else hyperparameters["units"]
        )
    )
    activation = (
        hp.Choice("activation", ["sigmoid", "relu", "tanh", "selu", "softplus", "softsign"])
        if "activation" not in hyperparameters_keys
        else (
            hp.Choice("activation", hyperparameters["activation"])
            if isinstance(hyperparameters["activation"], list)
            else hyperparameters["activation"]
        )
    )
    optimizer = (
        hp.Choice("optimizer", ["sgd", "adam", "adadelta", "rmsprop", "adamax", "adagrad"])
        if "optimizer" not in hyperparameters_keys
        else (
            hp.Choice("optimizer", hyperparameters["optimizer"])
            if isinstance(hyperparameters["optimizer"], list)
            else hyperparameters["optimizer"]
        )
    )
    threshold = (
        hp.Float("threshold", min_value=0.1, max_value=0.9, sampling="log")
        if "threshold" not in hyperparameters_keys
        else (
            hp.Choice("threshold", hyperparameters["threshold"])
            if isinstance(hyperparameters["threshold"], list)
            else hyperparameters["threshold"]
        )
    )
    num_layers = (
        hp.Int("num_layers", min_value=1, max_value=10, step=1)
        if "num_layers" not in hyperparameters_keys
        else (
            hp.Choice("num_layers", hyperparameters["num_layers"])
            if isinstance(hyperparameters["num_layers"], list)
            else hyperparameters["num_layers"]
        )
    )
    dropout = (
        hp.Float("dropout", min_value=0.1, max_value=0.9, sampling="log")
        if "dropout" not in hyperparameters_keys
        else (
            hp.Choice("dropout", hyperparameters["dropout"])
            if isinstance(hyperparameters["dropout"], list)
            else hyperparameters["dropout"]
        )
    )
    l2_reg = (
        hp.Float("l2_reg", min_value=1e-6, max_value=0.1, sampling="log")
        if "l2_reg" not in hyperparameters_keys
        else (
            hp.Choice("l2_reg", hyperparameters["l2_reg"])
            if isinstance(hyperparameters["l2_reg"], list)
            else hyperparameters["l2_reg"]
        )
    )

    model = call_existing_code(
        units=units,
        activation=activation,
        threshold=threshold,
        optimizer=optimizer,
        input_shape_parm=input_shape_parm,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=dropout,
        l2_reg=l2_reg,
    )
    return model


@suppress_warnings
def setup_model(
    data: DataFrame,
    target: str,
    epochs: int,
    train_size: float = 0.7,
    seed=None,
    train_mode: bool = True,
    filepath: str = "./my_dir/best_model",
    method: str = "Hyperband",
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
    method : `str`
        The method to use for hyperparameter tuning. Options are "Hyperband" and "RandomSearch".

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
    hyperparameters : `dict`
        The hyperparameters to set.

    Returns
    -------
    model : `AutoClassifier`
        The trained model.
    """
    max_trials = kwargs.get("max_trials", 10)
    directory = kwargs.get("directory", "./my_dir")
    project_name = kwargs.get("project_name", "get_best")
    objective = kwargs.get("objective", "val_loss")
    verbose = kwargs.get("verbose", True)
    hyperparameters = kwargs.get("hyperparameters", None)

    X = data.drop(columns=target)
    input_sample = X.sample(1)
    y = data[target]
    assert (
        X.select_dtypes(include=["object"]).empty == True
    ), "Categorical variables within the DataFrame must be encoded, this is done by using the DataFrameEncoder from likelihood."
    validation_split = 1.0 - train_size

    if train_mode:
        try:
            if (not os.path.exists(directory)) and directory != "./":
                os.makedirs(directory)
            elif directory != "./":
                print(f"Directory {directory} already exists, it will be deleted.")
                rmtree(directory)
                os.makedirs(directory)
        except:
            print("Warning: unable to create directory")

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
            build_model,
            input_shape_parm=input_shape_parm,
            num_classes=num_classes,
            hyperparameters=hyperparameters,
        )

        if method == "Hyperband":
            tuner = keras_tuner.Hyperband(
                hypermodel=build_model,
                objective=objective,
                max_epochs=epochs,
                factor=3,
                directory=directory,
                project_name=project_name,
                seed=seed,
            )
        elif method == "RandomSearch":
            tuner = keras_tuner.RandomSearch(
                hypermodel=build_model,
                objective=objective,
                max_trials=max_trials,
                directory=directory,
                project_name=project_name,
                seed=seed,
            )

        tuner.search(X, y, epochs=epochs, validation_split=validation_split, verbose=verbose)
        models = tuner.get_best_models(num_models=2)
        best_model = models[0]
        best_model(input_sample)

        best_model.save(filepath, save_format="tf")

        if verbose:
            tuner.results_summary()
    else:
        best_model = tf.keras.models.load_model(filepath)

    best_hps = tuner.get_best_hyperparameters(1)[0].values
    return best_model, pd.DataFrame(best_hps, index=["Value"])


class GetInsights:
    def __init__(self, model: AutoClassifier, inputs: np.ndarray) -> None:
        self.inputs = inputs
        self.model = model
        if isinstance(self.model.encoder.layers[0], InputLayer):
            self.encoder_layer = self.model.encoder.layers[1]
        else:
            self.encoder_layer = self.model.encoder.layers[0]
        self.decoder_layer = self.model.decoder.layers[0]
        self.encoder_weights = self.encoder_layer.get_weights()[0]
        self.decoder_weights = self.decoder_layer.get_weights()[0]
        colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

        by_hsv = sorted(
            (tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
            for name, color in colors.items()
        )
        self.sorted_names = [name for hsv, name in by_hsv if hsv[1] > 0.4 and hsv[2] >= 0.4]
        random.shuffle(self.sorted_names)

    def predictor_analyzer(
        self,
        frac=None,
        cmap: str = "viridis",
        aspect: str = "auto",
        highlight: bool = True,
        **kwargs,
    ) -> None:
        self._viz_weights(cmap=cmap, aspect=aspect, highlight=highlight, **kwargs)
        inputs = self.inputs.copy()
        y_labels = kwargs.get("y_labels", None)
        if frac:
            n = int(frac * self.inputs.shape[0])
            indexes = np.random.choice(np.arange(inputs.shape[0]), n, replace=False)
            inputs = inputs[indexes]
        inputs[np.isnan(inputs)] = 0.0
        # check if self.model.encoder(inputs) has two outputs
        try:
            mean, log_var = self.model.encoder(inputs)
            encoded = sampling(mean, log_var)
        except:
            encoded = self.model.encoder(inputs)
        reconstructed = self.model.decoder(encoded)
        combined = tf.concat([reconstructed, encoded], axis=1)
        self.classification = self.model.classifier(combined).numpy().argmax(axis=1)
        ax = plt.subplot(1, 2, 1)
        plt.imshow(self.inputs, cmap=cmap, aspect=aspect)
        plt.colorbar()
        plt.title("Original Data")
        plt.subplot(1, 2, 2, sharex=ax, sharey=ax)
        plt.imshow(reconstructed, cmap=cmap, aspect=aspect)
        plt.colorbar()
        plt.title("Decoder Layer Reconstruction")
        plt.show()

        self.data = pd.DataFrame(encoded, columns=[f"Feature {i}" for i in range(encoded.shape[1])])
        self.data_input = pd.DataFrame(
            inputs,
            columns=(
                [f"Feature {i}" for i in range(inputs.shape[1])] if y_labels is None else y_labels
            ),
        )

        self.data["class"] = self.classification
        self.data_input["class"] = self.classification

        try:
            self._get_tsne_repr(inputs=inputs, frac=frac)
            self._viz_tsne_repr(c=self.classification)

            self.data_normalized = self.data.copy(deep=True)
            self.data_normalized.iloc[:, :-1] = (
                2.0
                * (self.data_normalized.iloc[:, :-1] - self.data_normalized.iloc[:, :-1].min())
                / (
                    self.data_normalized.iloc[:, :-1].max()
                    - self.data_normalized.iloc[:, :-1].min()
                )
                - 1
            )
            radviz(self.data_normalized, "class", color=self.colors)
            plt.title("Radviz Visualization of Latent Space")
            plt.show()
            self.data_input_normalized = self.data_input.copy(deep=True)
            self.data_input_normalized.iloc[:, :-1] = (
                2.0
                * (
                    self.data_input_normalized.iloc[:, :-1]
                    - self.data_input_normalized.iloc[:, :-1].min()
                )
                / (
                    self.data_input_normalized.iloc[:, :-1].max()
                    - self.data_input_normalized.iloc[:, :-1].min()
                )
                - 1
            )
            radviz(self.data_input_normalized, "class", color=self.colors)
            plt.title("Radviz Visualization of Input Data")
            plt.show()
        except ValueError as e:
            plt.title("t-SNE Visualization of Latent Space")
            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            plt.show()
            warnings.warn(
                "Some functions or processes will not be executed for regression problems.",
                UserWarning,
            )
        return self._statistics(self.data_input)

    def _statistics(self, data_input: DataFrame, **kwargs) -> DataFrame:
        data = data_input.copy(deep=True)

        if not pd.api.types.is_string_dtype(data["class"]):
            data["class"] = data["class"].astype(str)

        data.ffill(inplace=True)
        grouped_data = data.groupby("class")

        numerical_stats = grouped_data.agg(["mean", "min", "max", "std", "median"])
        numerical_stats.columns = ["_".join(col).strip() for col in numerical_stats.columns.values]

        def get_mode(x):
            mode_series = x.mode()
            return mode_series.iloc[0] if not mode_series.empty else None

        mode_stats = grouped_data.apply(get_mode, include_groups=False)
        mode_stats.columns = [f"{col}_mode" for col in mode_stats.columns]
        combined_stats = pd.concat([numerical_stats, mode_stats], axis=1)

        return combined_stats.T

    def _viz_weights(
        self, cmap: str = "viridis", aspect: str = "auto", highlight: bool = True, **kwargs
    ) -> None:
        title = kwargs.get("title", "Encoder Layer Weights (Dense Layer)")
        y_labels = kwargs.get("y_labels", None)
        cmap_highlight = kwargs.get("cmap_highlight", "Pastel1")
        highlight_mask = np.zeros_like(self.encoder_weights, dtype=bool)

        plt.imshow(self.encoder_weights, cmap=cmap, aspect=aspect)
        plt.colorbar()
        plt.title(title)
        if y_labels is not None:
            plt.yticks(ticks=np.arange(self.encoder_weights.shape[0]), labels=y_labels)
        if highlight:
            for i, j in enumerate(self.encoder_weights.argmax(axis=1)):
                highlight_mask[i, j] = True
            plt.imshow(
                np.ma.masked_where(~highlight_mask, self.encoder_weights),
                cmap=cmap_highlight,
                alpha=0.5,
                aspect=aspect,
            )
        plt.show()

    def _get_tsne_repr(self, inputs=None, frac=None) -> None:
        if inputs is None:
            inputs = self.inputs.copy()
            if frac:
                n = int(frac * self.inputs.shape[0])
                indexes = np.random.choice(np.arange(inputs.shape[0]), n, replace=False)
                inputs = inputs[indexes]
            inputs[np.isnan(inputs)] = 0.0
        self.latent_representations = inputs @ self.encoder_weights

        tsne = TSNE(n_components=2)
        self.reduced_data_tsne = tsne.fit_transform(self.latent_representations)

    def _viz_tsne_repr(self, **kwargs) -> None:
        c = kwargs.get("c", None)
        self.colors = (
            kwargs.get("colors", self.sorted_names[: len(np.unique(c))]) if c is not None else None
        )
        plt.scatter(
            self.reduced_data_tsne[:, 0],
            self.reduced_data_tsne[:, 1],
            cmap=matplotlib.colors.ListedColormap(self.colors) if c is not None else None,
            c=c,
        )
        if c is not None:
            cb = plt.colorbar()
            loc = np.arange(0, max(c), max(c) / float(len(self.colors)))
            cb.set_ticks(loc)
            cb.set_ticklabels(np.unique(c))
        plt.title("t-SNE Visualization of Latent Space")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.show()


########################################################################################

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import OneHotEncoder

    # Load the dataset
    iris = load_iris()

    # Convert to a DataFrame for easy exploration
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df["species"] = iris.target

    X = iris_df.drop(columns="species")
    y_labels = X.columns
    X = X.values
    y = iris_df["species"].values

    X = np.asarray(X).astype(np.float32)

    encoder = OneHotEncoder()
    y = encoder.fit_transform(y.reshape(-1, 1)).toarray()
    y = np.asarray(y).astype(np.float32)

    model = AutoClassifier(
        input_shape_parm=X.shape[1],
        num_classes=3,
        units=27,
        activation="tanh",
        num_layers=2,
        dropout=0.2,
    )
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.F1Score(threshold=0.5)],
    )
    model.fit(X, y, epochs=50, validation_split=0.2)

    insights = GetInsights(model, X)
    summary = insights.predictor_analyzer(frac=1.0, y_labels=y_labels)
    insights._get_tsne_repr()
    insights._viz_tsne_repr()
    insights._viz_tsne_repr(c=iris_df["species"])
    insights._viz_weights()
    print(summary)
