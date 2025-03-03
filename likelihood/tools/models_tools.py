import logging
import os

import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)


@tf.keras.utils.register_keras_serializable(package="Custom", name="LoRALayer")
class LoRALayer(tf.keras.layers.Layer):
    def __init__(self, units, rank=4, **kwargs):
        super(LoRALayer, self).__init__(**kwargs)
        self.units = units
        self.rank = rank

    def build(self, input_shape):
        input_dim = input_shape[-1]
        print(f"Input shape: {input_shape}")

        self.A = self.add_weight(
            shape=(input_dim, self.rank), initializer="random_normal", trainable=True, name="A"
        )
        self.B = self.add_weight(
            shape=(self.rank, self.units), initializer="random_normal", trainable=True, name="B"
        )
        print(f"Dense weights shape: {input_dim}x{self.units}")
        print(f"LoRA weights shape: A{self.A.shape}, B{self.B.shape}")

    def call(self, inputs):
        lora_output = tf.matmul(tf.matmul(inputs, self.A), self.B)
        return lora_output


def apply_lora(model, rank=4):
    inputs = tf.keras.Input(shape=model.input_shape[1:])
    x = inputs

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            print(f"Applying LoRA to layer {layer.name}")
            x = LoRALayer(units=layer.units, rank=rank)(x)
        else:
            x = layer(x)
    new_model = tf.keras.Model(inputs=inputs, outputs=x)
    return new_model
