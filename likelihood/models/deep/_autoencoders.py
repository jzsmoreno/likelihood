from .autoencoders import (
    DataFrame,
    EarlyStopping,
    LoRALayer,
    OneHotEncoder,
    cal_loss_step,
    keras_tuner,
    l2,
    np,
    partial,
    pd,
    sampling,
    suppress_warnings,
    tf,
    train_step,
)


@tf.keras.utils.register_keras_serializable(package="Custom", name="stabilize_log_var")
def stabilize_log_var(x):
    return x + 1e-7


@tf.keras.utils.register_keras_serializable(package="Custom", name="sampling_wrapper")
def sampling_wrapper(args):
    mean, log_var = args
    return sampling(mean, log_var)


@tf.keras.utils.register_keras_serializable(package="Custom", name="sampling_output_shape")
def sampling_output_shape(input_shapes):
    return input_shapes[0]


class AutoClassifier:
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
        The activation function to use for the classifier layer. Default is `softmax`. If the activation function is not a classification function, the model can be used in regression problems.
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
        self.input_shape_parm = input_shape_parm
        self.num_classes = num_classes
        self.units = units
        self.activation = activation

        # Store all configuration parameters
        self.classifier_activation = kwargs.get("classifier_activation", "softmax")
        self.num_layers = kwargs.get("num_layers", 1)
        self.dropout = kwargs.get("dropout", None)
        self.l2_reg = kwargs.get("l2_reg", 0.0)
        self.vae_mode = kwargs.get("vae_mode", False)
        self.vae_units = kwargs.get("vae_units", 2)
        self.lora_mode = kwargs.get("lora_mode", False)
        self.lora_rank = kwargs.get("lora_rank", 4)

        # Initialize models as None - will be built when needed
        self._encoder = None
        self._decoder = None
        self._classifier = None
        self._main_model = None

        # Build all models
        self._build_models()

    def _build_encoder(self):
        """Build the encoder model."""
        if self.vae_mode:
            inputs = tf.keras.Input(shape=(self.input_shape_parm,), name="encoder_input")
            x = tf.keras.layers.Dense(
                units=self.units,
                kernel_regularizer=l2(self.l2_reg),
                kernel_initializer="he_normal",
                name="vae_encoder_dense_1",
            )(inputs)
            x = tf.keras.layers.BatchNormalization(name="vae_encoder_bn_1")(x)
            x = tf.keras.layers.Activation(self.activation, name="vae_encoder_act_1")(x)
            x = tf.keras.layers.Dense(
                units=int(self.units / 2),
                kernel_regularizer=l2(self.l2_reg),
                kernel_initializer="he_normal",
                name="encoder_hidden",
            )(x)
            x = tf.keras.layers.BatchNormalization(name="vae_encoder_bn_2")(x)
            x = tf.keras.layers.Activation(self.activation, name="vae_encoder_act_2")(x)

            mean = tf.keras.layers.Dense(self.vae_units, name="mean")(x)
            log_var = tf.keras.layers.Dense(self.vae_units, name="log_var")(x)
            log_var = tf.keras.layers.Lambda(stabilize_log_var, name="log_var_stabilized")(log_var)

            self._encoder = tf.keras.Model(inputs, [mean, log_var], name="vae_encoder")
        else:
            inputs = tf.keras.Input(shape=(self.input_shape_parm,), name="encoder_input")
            x = tf.keras.layers.Dense(
                units=self.units,
                activation=self.activation,
                kernel_regularizer=l2(self.l2_reg),
                name="encoder_dense_1",
            )(inputs)
            outputs = tf.keras.layers.Dense(
                units=int(self.units / 2),
                activation=self.activation,
                kernel_regularizer=l2(self.l2_reg),
                name="encoder_dense_2",
            )(x)

            self._encoder = tf.keras.Model(inputs, outputs, name="encoder")

    def _build_decoder(self):
        """Build the decoder model."""
        if self.vae_mode:
            inputs = tf.keras.Input(shape=(self.vae_units,), name="decoder_input")
            x = tf.keras.layers.Dense(
                units=self.units, kernel_regularizer=l2(self.l2_reg), name="vae_decoder_dense_1"
            )(inputs)
            x = tf.keras.layers.BatchNormalization(name="vae_decoder_bn_1")(x)
            x = tf.keras.layers.Activation(self.activation, name="vae_decoder_act_1")(x)
            x = tf.keras.layers.Dense(
                units=self.input_shape_parm,
                kernel_regularizer=l2(self.l2_reg),
                name="vae_decoder_dense_2",
            )(x)
            x = tf.keras.layers.BatchNormalization(name="vae_decoder_bn_2")(x)
            outputs = tf.keras.layers.Activation(self.activation, name="vae_decoder_act_2")(x)
        else:
            inputs = tf.keras.Input(shape=(int(self.units / 2),), name="decoder_input")
            x = tf.keras.layers.Dense(
                units=self.units,
                activation=self.activation,
                kernel_regularizer=l2(self.l2_reg),
                name="decoder_dense_1",
            )(inputs)
            outputs = tf.keras.layers.Dense(
                units=self.input_shape_parm,
                activation=self.activation,
                kernel_regularizer=l2(self.l2_reg),
                name="decoder_dense_2",
            )(x)

        self._decoder = tf.keras.Model(inputs, outputs, name="decoder")

    def _build_classifier(self):
        """Build the classifier model."""
        # Input shape is decoded + encoded features
        if self.vae_mode:
            input_dim = self.input_shape_parm + self.vae_units
        else:
            input_dim = self.input_shape_parm + int(self.units / 2)

        inputs = tf.keras.Input(shape=(input_dim,), name="classifier_input")
        x = inputs

        # Build hidden layers
        if self.num_layers > 1 and not self.lora_mode:
            for i in range(self.num_layers - 1):
                x = tf.keras.layers.Dense(
                    units=self.units,
                    activation=self.activation,
                    kernel_regularizer=l2(self.l2_reg),
                    name=f"classifier_dense_{i+1}",
                )(x)
                if self.dropout:
                    x = tf.keras.layers.Dropout(self.dropout, name=f"classifier_dropout_{i+1}")(x)

        elif self.lora_mode and self.num_layers > 1:
            for i in range(self.num_layers - 1):
                x = LoRALayer(units=self.units, rank=self.lora_rank, name=f"LoRA_{i}")(x)
                x = tf.keras.layers.Activation(self.activation, name=f"lora_activation_{i+1}")(x)
                if self.dropout:
                    x = tf.keras.layers.Dropout(self.dropout, name=f"lora_dropout_{i+1}")(x)

        # Output layer
        outputs = tf.keras.layers.Dense(
            units=self.num_classes,
            activation=self.classifier_activation,
            kernel_regularizer=l2(self.l2_reg),
            name="classifier_output",
        )(x)

        self._classifier = tf.keras.Model(inputs, outputs, name="classifier")

    def _build_main_model(self):
        """Build the main model that combines encoder, decoder, and classifier."""
        inputs = tf.keras.Input(shape=(self.input_shape_parm,), name="main_input")

        # Encoder forward pass
        if self.vae_mode:
            mean, log_var = self._encoder(inputs)
            # Sampling layer
            encoded = tf.keras.layers.Lambda(
                sampling_wrapper, output_shape=sampling_output_shape, name="sampling_layer"
            )([mean, log_var])
        else:
            encoded = self._encoder(inputs)

        # Decoder forward pass
        decoded = self._decoder(encoded)

        # Combine decoded and encoded features
        combined = tf.keras.layers.Concatenate(name="combine_features")([decoded, encoded])

        # Classifier forward pass
        outputs = self._classifier(combined)

        self._main_model = tf.keras.Model(
            inputs=inputs, outputs=outputs, name="auto_classifier_main"
        )

    def _build_models(self):
        """Build all component models."""
        self._build_encoder()
        self._build_decoder()
        self._build_classifier()
        self._build_main_model()

    @property
    def encoder(self):
        """Get the encoder model."""
        return self._encoder

    @encoder.setter
    def encoder(self, value):
        """Set the encoder model and rebuild main model."""
        self._encoder = value
        if self._decoder and self._classifier:
            self._build_main_model()

    @property
    def decoder(self):
        """Get the decoder model."""
        return self._decoder

    @decoder.setter
    def decoder(self, value):
        """Set the decoder model and rebuild main model."""
        self._decoder = value
        if self._encoder and self._classifier:
            self._build_main_model()

    @property
    def classifier(self):
        """Get the classifier model."""
        return self._classifier

    @classifier.setter
    def classifier(self, value):
        """Set the classifier model and rebuild main model."""
        self._classifier = value
        if self._encoder and self._decoder:
            self._build_main_model()

    def train_encoder_decoder(
        self, data, epochs, batch_size, validation_split=0.2, patience=10, **kwargs
    ):
        """
        Trains the encoder and decoder on the input data.

        Parameters
        ----------
        data : tf.data.Dataset, np.ndarray
            The input data.
        epochs : int
            The number of epochs to train for.
        batch_size : int
            The batch size to use.
        validation_split : float
            The proportion of the dataset to use for validation. Default is 0.2.
        patience : int
            The number of epochs to wait before early stopping. Default is 10.
        """
        verbose = kwargs.get("verbose", True)
        optimizer = kwargs.get("optimizer", tf.keras.optimizers.Adam())

        # Prepare data
        if isinstance(data, np.ndarray):
            data = tf.data.Dataset.from_tensor_slices(data).batch(batch_size)
            data = data.map(lambda x: tf.cast(x, tf.float32))

        early_stopping = EarlyStopping(patience=patience)
        train_batches = data.take(int((1 - validation_split) * len(data)))
        val_batches = data.skip(int((1 - validation_split) * len(data)))

        for epoch in range(epochs):
            train_loss = 0
            val_loss = 0

            # Training step
            for train_batch in train_batches:
                loss_train = train_step(
                    train_batch, self._encoder, self._decoder, optimizer, self.vae_mode
                )
                train_loss = loss_train  # Keep last batch loss

            # Validation step
            for val_batch in val_batches:
                loss_val = cal_loss_step(
                    val_batch, self._encoder, self._decoder, self.vae_mode, False
                )
                val_loss = loss_val  # Keep last batch loss

            early_stopping(train_loss)

            if early_stopping.stop_training:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch}.")
                break

            if epoch % 10 == 0 and verbose:
                print(
                    f"Epoch {epoch}: Train Loss: {train_loss:.6f} Validation Loss: {val_loss:.6f}"
                )

        self.freeze_encoder_decoder()

    def freeze_encoder_decoder(self):
        """Freezes the encoder and decoder layers to prevent them from being updated during training."""
        if self._encoder:
            for layer in self._encoder.layers:
                layer.trainable = False
        if self._decoder:
            for layer in self._decoder.layers:
                layer.trainable = False

        # Rebuild main model to reflect trainability changes
        self._build_main_model()

    def unfreeze_encoder_decoder(self):
        """Unfreezes the encoder and decoder layers allowing them to be updated during training."""
        if self._encoder:
            for layer in self._encoder.layers:
                layer.trainable = True
        if self._decoder:
            for layer in self._decoder.layers:
                layer.trainable = True

        # Rebuild main model to reflect trainability changes
        self._build_main_model()

    def set_encoder_decoder(self, source_model):
        """
        Sets the encoder and decoder layers from another AutoClassifier instance,
        ensuring compatibility in dimensions.

        Parameters
        ----------
        source_model : AutoClassifier
            The source model to copy the encoder and decoder layers from.

        Raises
        ------
        ValueError
            If the input shape or units of the source model do not match.
        """
        if not isinstance(source_model, AutoClassifier):
            raise ValueError("Source model must be an instance of AutoClassifier.")

        if self.input_shape_parm != source_model.input_shape_parm:
            raise ValueError(
                f"Incompatible input shape. Expected {self.input_shape_parm}, got {source_model.input_shape_parm}."
            )
        if self.units != source_model.units:
            raise ValueError(
                f"Incompatible number of units. Expected {self.units}, got {source_model.units}."
            )

        # Clone and copy weights
        if source_model._encoder:
            self._encoder = tf.keras.models.clone_model(source_model._encoder)
            self._encoder.set_weights(source_model._encoder.get_weights())

        if source_model._decoder:
            self._decoder = tf.keras.models.clone_model(source_model._decoder)
            self._decoder.set_weights(source_model._decoder.get_weights())

        # Rebuild main model with new encoder/decoder
        self._build_main_model()

    # Main model interface methods
    def __call__(self, x, training=None):
        """Forward pass through the model."""
        return self._main_model(x, training=training)

    def compile(self, *args, **kwargs):
        """Compile the main model."""
        return self._main_model.compile(*args, **kwargs)

    def fit(self, *args, **kwargs):
        """Fit the main model."""
        return self._main_model.fit(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        """Evaluate the main model."""
        return self._main_model.evaluate(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """Predict using the main model."""
        return self._main_model.predict(*args, **kwargs)

    def save(self, filepath, **kwargs):
        """
        Save the complete model including all components.

        Parameters
        ----------
        filepath : str
            Path where to save the model.
        """
        import os

        # Create directory if it doesn't exist
        os.makedirs(filepath, exist_ok=True)

        # Save all component models
        self._encoder.save(os.path.join(filepath, "encoder.keras"))
        self._decoder.save(os.path.join(filepath, "decoder.keras"))
        self._classifier.save(os.path.join(filepath, "classifier.keras"))
        self._main_model.save(os.path.join(filepath, "main_model.keras"))

        # Save configuration
        import json

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

        with open(os.path.join(filepath, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, filepath):
        """
        Load a complete model from saved components.

        Parameters
        ----------
        filepath : str
            Path where the model was saved.

        Returns
        -------
        AutoClassifier
            The loaded model instance.
        """
        import json
        import os

        # Load configuration
        with open(os.path.join(filepath, "config.json"), "r") as f:
            config = json.load(f)

        # Create new instance
        instance = cls(**config)

        # Load component models
        instance._encoder = tf.keras.models.load_model(os.path.join(filepath, "encoder.keras"))
        instance._decoder = tf.keras.models.load_model(os.path.join(filepath, "decoder.keras"))
        instance._classifier = tf.keras.models.load_model(
            os.path.join(filepath, "classifier.keras")
        )
        instance._main_model = tf.keras.models.load_model(
            os.path.join(filepath, "main_model.keras")
        )

        return instance

    # Additional properties and methods for compatibility
    @property
    def weights(self):
        """Get all model weights."""
        return self._main_model.weights

    def get_weights(self):
        """Get all model weights."""
        return self._main_model.get_weights()

    def set_weights(self, weights):
        """Set all model weights."""
        return self._main_model.set_weights(weights)

    @property
    def trainable_variables(self):
        """Get trainable variables."""
        return self._main_model.trainable_variables

    @property
    def non_trainable_variables(self):
        """Get non-trainable variables."""
        return self._main_model.non_trainable_variables

    def summary(self, *args, **kwargs):
        """Print model summary."""
        print("=== AutoClassifier Summary ===")
        print("\n--- Encoder ---")
        self._encoder.summary(*args, **kwargs)
        print("\n--- Decoder ---")
        self._decoder.summary(*args, **kwargs)
        print("\n--- Classifier ---")
        self._classifier.summary(*args, **kwargs)
        print("\n--- Main Model ---")
        self._main_model.summary(*args, **kwargs)

    def get_config(self):
        """Get model configuration."""
        return {
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

    Keyword Arguments:
    ----------
    vae_mode : `bool`
        Whether to use variational autoencoder mode. Default is False.
    vae_units : `int`
        The number of units in the variational autoencoder. Default is 2.

    Returns
    -------
    `AutoClassifier`
        The AutoClassifier instance.
    """
    dropout = kwargs.get("dropout", None)
    l2_reg = kwargs.get("l2_reg", 0.0)
    vae_mode = kwargs.get("vae_mode", False)
    vae_units = kwargs.get("vae_units", 2)
    model = AutoClassifier(
        input_shape_parm=input_shape_parm,
        num_classes=num_classes,
        units=units,
        activation=activation,
        num_layers=num_layers,
        dropout=dropout,
        l2_reg=l2_reg,
        vae_mode=vae_mode,
        vae_units=vae_units,
    )
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.F1Score(threshold=threshold)],
    )
    return model._main_model


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
    vae_mode = (
        hp.Choice("vae_mode", [True, False])
        if "vae_mode" not in hyperparameters_keys
        else hyperparameters["vae_mode"]
    )

    try:
        vae_units = (
            hp.Int("vae_units", min_value=2, max_value=10, step=1)
            if ("vae_units" not in hyperparameters_keys) and vae_mode
            else (
                hp.Choice("vae_units", hyperparameters["vae_units"])
                if isinstance(hyperparameters["vae_units"], list)
                else hyperparameters["vae_units"]
            )
        )
    except KeyError:
        vae_units = None

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
        vae_mode=vae_mode,
        vae_units=vae_units,
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

        best_model.save(filepath if filepath.endswith(".keras") else filepath + ".keras")

        if verbose:
            tuner.results_summary()
    else:
        best_model = tf.keras.models.load_model(
            filepath if filepath.endswith(".keras") else filepath + ".keras"
        )
    best_hps = tuner.get_best_hyperparameters(1)[0].values
    vae_mode = best_hps.get("vae_mode", hyperparameters.get("vae_mode", False))
    best_hps["vae_units"] = None if not vae_mode else best_hps["vae_units"]

    return best_model, pd.DataFrame(best_hps, index=["Value"]).dropna(axis=1)
