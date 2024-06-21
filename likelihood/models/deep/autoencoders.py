from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model


class AutoClassifier(Model):
    def __init__(self, input_shape, latent_dim, num_classes):
        super(AutoClassifier, self).__init__()
        self.latent_dim = latent_dim
        self.shape = input_shape

        self.encoder = tf.keras.Sequential(
            [tf.keras.layers.Dense(latent_dim, activation="sigmoid")]
        )

        self.decoder = tf.keras.Sequential(
            [tf.keras.layers.Dense(input_shape[1], activation="sigmoid")]
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
