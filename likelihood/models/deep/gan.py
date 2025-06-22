import numpy as np
import pandas as pd
import tensorflow as tf
from IPython.display import clear_output
from tensorflow.keras.regularizers import l2
from tqdm import tqdm

from likelihood.tools import get_metrics


@tf.keras.utils.register_keras_serializable(package="Custom", name="GANRegressor")
class GANRegressor(tf.keras.Model):
    """
    GANRegressor is a custom Keras model that combines a generator and a discriminator
    """

    def __init__(
        self,
        input_shape_parm,
        output_shape_parm,
        num_neurons=128,
        activation="linear",
        depth=5,
        dropout=0.2,
        l2_reg=0.0,
        **kwargs,
    ):
        super(GANRegressor, self).__init__()
        self.input_shape_parm = input_shape_parm
        self.output_shape_parm = output_shape_parm
        self.num_neurons = num_neurons
        self.activation = activation
        self.depth = depth
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.optimizer = kwargs.get("optimizer", "adam")

        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        dummy_input = tf.convert_to_tensor(tf.random.normal([1, self.input_shape_parm]))
        self.build(dummy_input.shape)

    def build(self, input_shape):
        self.gan = tf.keras.models.Sequential([self.generator, self.discriminator], name="gan")

        self.generator.compile(
            optimizer=self.optimizer,
            loss=tf.keras.losses.MeanAbsolutePercentageError(),
            metrics=[tf.keras.metrics.MeanAbsolutePercentageError()],
        )

        self.discriminator.compile(
            optimizer=self.optimizer, loss="binary_crossentropy", metrics=["accuracy"]
        )

        self.gan.compile(optimizer=self.optimizer, loss="binary_crossentropy")
        super(GANRegressor, self).build(input_shape)

    def _build_generator(self):
        generator = tf.keras.Sequential(name="generator")
        generator.add(
            tf.keras.layers.Dense(
                self.num_neurons,
                activation="selu",
                input_shape=[self.input_shape_parm],
                kernel_regularizer=l2(self.l2_reg),
            )
        )
        generator.add(tf.keras.layers.Dropout(self.dropout))
        for _ in range(self.depth - 1):
            generator.add(
                tf.keras.layers.Dense(
                    self.num_neurons, activation="selu", kernel_regularizer=l2(self.l2_reg)
                ),
            )
            generator.add(tf.keras.layers.Dropout(self.dropout))
        generator.add(tf.keras.layers.Dense(2 * self.output_shape_parm, activation=self.activation))
        return generator

    def _build_discriminator(self):
        discriminator = tf.keras.Sequential(name="discriminator")
        for _ in range(self.depth):
            discriminator.add(
                tf.keras.layers.Dense(
                    self.num_neurons, activation="selu", kernel_regularizer=l2(self.l2_reg)
                ),
            )
            discriminator.add(tf.keras.layers.Dropout(self.dropout))
        discriminator.add(tf.keras.layers.Dense(2, activation="softmax"))
        return discriminator

    def train_gan(
        self,
        X,
        y,
        batch_size,
        n_epochs,
        validation_split=0.2,
        verbose=1,
    ):
        """
        Train the GAN model.

        Parameters
        --------
        X : array-like
            Input data.
        y : array-like
            Target data.
        batch_size : int
            Number of samples in each batch.
        n_epochs : int
            Number of training epochs.
        validation_split : float, optional
            Fraction of the data to be used for validation.
        verbose : int, optional
            Verbosity level. Default is 1.

        Returns
        --------
        history : pd.DataFrame
            Training history.
        """
        loss_history = []
        for epoch in tqdm(range(n_epochs)):
            batch_starts = np.arange(0, len(X), batch_size)
            for start in batch_starts:
                np.random.shuffle(batch_starts)
                end = start + batch_size
                X_batch = X[start:end]
                y_batch = y[start:end].reshape(-1, self.output_shape_parm)
                y_batch = np.concatenate((y_batch, y_batch**2), axis=1)
                X_batch = tf.cast(X_batch, "float32")
                noise = tf.random.normal(
                    shape=X_batch.shape, stddev=tf.math.reduce_std(X_batch, keepdims=False)
                )

                # Phase 1 - training the generator
                self.generator.train_on_batch(X_batch, y_batch)

                # Phase 2 - training the discriminator
                generated_y_fake = self.generator(noise)
                generated_y_real = self.generator(X_batch)
                fake_and_real = tf.concat([generated_y_fake, generated_y_real], axis=0)
                X_fake_and_real = tf.concat([noise, X_batch], axis=0)
                batch_size = int(fake_and_real.shape[0] / 2)
                indices_ = tf.constant([[0.0]] * batch_size + [[1.0]] * batch_size)[:, 0]
                indices_ = tf.cast(indices_, "int32")
                y1 = tf.one_hot(indices_, 2)
                self.gan.train_on_batch(X_fake_and_real, y1)

            loss = self._cal_loss(generated_y_real, y_batch)
            loss_history.append([epoch, loss])

        if verbose:
            X_batch, y_batch, X_batch_val, y_batch_val = self._train_and_val(
                X_batch, y_batch, validation_split=validation_split
            )
            generated_y = self.generator(X_batch)
            generated_y_val = self.generator(X_batch_val)
            y_pred = self.discriminator.predict(fake_and_real, verbose=0)
            y_pred = list(np.argmax(y_pred, axis=1))

            metrics = get_metrics(self._get_frame(indices_.numpy().tolist(), y_pred), "y", "y_pred")
            loss = self._cal_loss(generated_y, y_batch)
            loss_val = self._cal_loss(generated_y_val, y_batch_val)
            clear_output(wait=True)
            metrics_list = [
                ("Epoch", f"{epoch}"),
                ("Loss", f"{loss:.2f} / {loss_val:.2f}"),
                ("Accuracy", f"{metrics['accuracy']:.2f} / {metrics['accuracy']:.2f}"),
                ("Precision", f"{metrics['precision']:.2f} / {metrics['precision']:.2f}"),
                ("Recall", f"{metrics['recall']:.2f} / {metrics['recall']:.2f}"),
                ("F1 Score", f"{metrics['f1_score']:.2f} / {metrics['f1_score']:.2f}"),
                ("Kappa", f"{metrics['kappa']:.2f} / {metrics['kappa']:.2f}"),
            ]

            metric_width = 15
            value_width = 30

            header = f"| {'Metric':<{metric_width}} | {'Value':<{value_width}} |"
            separator = "+" + "-" * (len(header) - 2) + "+"

            print(separator)
            print(header)
            print(separator)

            for metric_name, metric_values in metrics_list:
                data_row = f"| {metric_name:<{metric_width}} | {metric_values:<{value_width}} |"
                print(data_row)

            print(separator)

        return pd.DataFrame(loss_history, columns=["epoch", "loss"])

    def _get_frame(self, y, y_pred):
        df = pd.DataFrame()
        df["y"] = y
        df["y_pred"] = y_pred
        return df

    def _train_and_val(self, X, y, validation_split):
        split = int((1 - validation_split) * len(X))

        if len(X) > split and split > 0:
            X_train = X[:split]
            y_train = y[:split]
            X_val = X[split:]
            y_val = y[split:]
        else:
            X_train = X
            y_train = y
            X_val = X
            y_val = y

        X_train = tf.cast(X_train, "float32")
        X_val = tf.cast(X_val, "float32")

        return X_train, y_train, X_val, y_val

    def _cal_loss(self, generated, y):
        return tf.math.reduce_mean(100 * abs((y - generated) / y), keepdims=False).numpy()

    def train_gen(
        self,
        X_train,
        y_train,
        batch_size,
        n_epochs,
        validation_split=0.2,
        patience=3,
    ):
        """
        Train the generator model.

        Parameters
        --------
        X_train : array-like
            Training data.
        y_train : array-like
            Training target data.
        batch_size : int
            Batch size for training.
        n_epochs : int
            Number of epochs for training.
        validation_split : float, optional
            Fraction of data to use for validation. Default is 0.2.
        patience : int, optional
            Number of epochs to wait before early stopping. Default is 3.

        Returns
        --------
        history : pd.DataFrame
            Training history.
        """
        callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience)
        # Prepare the target by extending it with its square
        self.discriminator.trainable = False
        y_train_extended = np.concatenate(
            (
                y_train.reshape(-1, self.output_shape_parm),
                y_train.reshape(-1, self.output_shape_parm) ** 2,
            ),
            axis=1,
        )

        history = self.generator.fit(
            X_train,
            y_train_extended,
            epochs=n_epochs,
            batch_size=batch_size,
            verbose=0,
            validation_split=validation_split,
            callbacks=[callback],
        )

        return pd.DataFrame(history.history)

    def call(self, inputs):
        return self.generator(inputs)[:, 0]

    def get_config(self):
        config = {
            "input_shape_parm": self.input_shape_parm,
            "output_shape_parm": self.output_shape_parm,
            "num_neurons": self.num_neurons,
            "activation": self.activation,
            "depth": self.depth,
            "dropout": self.dropout,
            "generator": self.generator,
            "discriminator": self.discriminator,
            "gan": self.gan,
            "l2_reg": self.l2_reg,
            "optimizer": self.optimizer,
        }
        base_config = super(GANRegressor, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(
            input_shape_parm=config["input_shape_parm"],
            output_shape_parm=config["output_shape_parm"],
            num_neurons=config["num_neurons"],
            activation=config["activation"],
            depth=config["depth"],
            dropout=config["dropout"],
            generator=config["generator"],
            discriminator=config["discriminator"],
            gan=config["gan"],
            l2_reg=config["l2_reg"],
            optimizer=config["optimizer"],
        )
