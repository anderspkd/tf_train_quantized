import tensorflow as tf
from tensorflow import keras

## MNIST models

def build_model_simple():
    return keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28, 1)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

def build_model_simple2():
    return keras.Sequential([
        keras.layers.Conv2D(16, (3, 3), input_shape=(28, 28, 1)),
        keras.layers.ReLU(max_value=6),
        keras.layers.Flatten(),
        keras.layers.Dense(256),
        keras.layers.ReLU(max_value=6),
        keras.layers.Dense(10, activation='softmax')
    ])

## Put models in the dict below to make them available to the training scripts.

models = {
    'mnist_simple1': build_model_simple,
    'mnist_simple2': build_model_simple2
}
