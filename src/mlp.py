"""A simple multilayer perceptron model."""
import tensorflow as tf
# pylint: disable=import-error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization
from tensorflow.keras.layers import Dropout

def mlp(hidden, nodes, input_size, dropout=0):
    """Multilayer perceptron: (Dense+Droupout+BatchNorm+ReLu)

    Args:
        hidden (int): number of hidden layers.
        nodes (int): number of nodes per layer.
        input_size (int): input size.
        dropout (float): dropout rate.

    Returns:
        model
    """
    model = Sequential()
    model.add(Dense(nodes, kernel_initializer='he_normal',
                    input_shape=(input_size,)))
    if dropout != 0:
        model.add(Dropout(dropout))
    model.add(BatchNormalization())
    model.add(Activation(tf.nn.relu))

    for _ in range(hidden):
        model.add(Dense(nodes, kernel_initializer='he_normal'))
        if dropout != 0:
            model.add(Dropout(dropout))
        model.add(BatchNormalization())
        model.add(Activation(tf.nn.relu))

    model.add(Dense(16, kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation(tf.nn.softmax))

    return model
