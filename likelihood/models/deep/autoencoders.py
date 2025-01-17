import logging
import os
import random
from functools import partial
from shutil import rmtree

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import radviz

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import keras_tuner
import tensorflow as tf
from pandas.core.frame import DataFrame
from sklearn.manifold import TSNE

from likelihood.tools import OneHotEncoder


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


class GetInsights:
    def __init__(self, model: AutoClassifier, inputs: np.ndarray) -> None:
        self.inputs = inputs
        self.model = model
        self.encoder_layer = self.model.encoder.layers[0]
        self.decoder_layer = self.model.decoder.layers[0]
        self.classifier_layer = self.model.classifier.layers[0]
        self.encoder_weights = self.encoder_layer.get_weights()[0]
        self.decoder_weights = self.decoder_layer.get_weights()[0]
        self.classifier_weights = self.classifier_layer.get_weights()[0]
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
        if frac:
            n = int(frac * self.inputs.shape[0])
            indexes = np.random.choice(np.arange(inputs.shape[0]), n, replace=False)
            inputs = inputs[indexes]
        inputs[np.isnan(inputs)] = 0.0
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

        self._get_tsne_repr(inputs=inputs, frac=frac)
        self._viz_tsne_repr(c=self.classification)

        data = pd.DataFrame(encoded, columns=[f"Feature {i}" for i in range(encoded.shape[1])])
        data_input = pd.DataFrame(
            inputs,
            columns=(
                [f"Feature {i}" for i in range(inputs.shape[1])] if y_labels is None else y_labels
            ),
        )
        data["class"] = self.classification
        data_input["class"] = self.classification
        radviz(data, "class", color=self.colors)
        plt.title("Radviz Visualization of Latent Space")
        plt.show()

        radviz(data_input, "class", color=self.colors)
        plt.title("Radviz Visualization of Input Data")
        plt.show()

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

    model = AutoClassifier(input_shape_parm=X.shape[1], num_classes=3, units=27, activation="selu")
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.F1Score(threshold=0.5)],
    )
    model.fit(X, y, epochs=50, validation_split=0.2)

    insights = GetInsights(model, X)
    insights.predictor_analyzer(frac=1.0, y_labels=y_labels)
    insights._get_tsne_repr()
    insights._viz_tsne_repr()
    insights._viz_tsne_repr(c=iris_df["species"])
    insights._viz_weights()
