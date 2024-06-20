from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.models import Model


class AutoClassifier(Model):
    def __init__(self, input_shape, latent_dim, num_classes):
        super(AutoClassifier, self).__init__()
        self.latent_dim = latent_dim
        self.shape = input_shape

        self.encoder = tf.keras.Sequential(
            [tf.keras.layers.Flatten(), tf.keras.layers.Dense(latent_dim, activation="sigmoid")]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    tf.math.reduce_prod(input_shape).numpy(), activation="sigmoid"
                ),
                tf.keras.layers.Reshape(input_shape),
            ]
        )

        self.classifier = tf.keras.Sequential(
            [tf.keras.layers.Dense(num_classes, activation="softmax")]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        combined = concatenate([tf.reshape(decoded, [-1]), encoded])
        classifier = self.classifier(combined)
        return classifier


if __name__ == "__main__":

    (x_train, _), (x_test, _) = fashion_mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    shape = x_test.shape[1:]
    latent_dim = 64
    num_classes = 10
    model = AutoClassifier(shape, latent_dim, num_classes)
    model.compile(optimizer="adam", loss=losses.MeanSquaredError())
