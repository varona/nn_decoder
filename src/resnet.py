"""Resnet model similar to: https://keras.io/examples/cifar10_resnet/
but no downsampling after each block, periodic padding implemented with a
Lambda layer (since the lattice is periodic) and no average pooling.
"""
import tensorflow as tf
# pylint: disable=import-error
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, add
from tensorflow.keras.layers import Input, Flatten, Lambda, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model


def periodic_padding(tensor,
                     axis=(1, 2),
                     padding=(1, 1),
                     one_sided=False):
    """Add periodic padding to a tensor for specified axis.

    Periodic boundary conditions for lattice as shown in Table II of 
    arXiv:2002.08666, thus we use tf.roll() for axis=1.

    Args:
        tensor: input tensor.
        axis (int or tuple): one or multiple axis to pad along, int or tuple.
        padding (int or tuple): number of cells to pad.
        one_sided (bool): if true pad only one of the sides.
        compact (bool): if true .

    Returns: 
        padded tensor.

    https://stackoverflow.com/questions/39088489/tensorflow-periodic-padding
    https://stackoverflow.com/questions/50677544/reflection-padding-conv2d
    """

    if isinstance(axis, int):
        axis = (axis,)
    if isinstance(padding, int):
        padding = (padding,)

    ndim = len(tensor.shape)
    for ax, p in zip(axis, padding):
        # create a slice object that selects everything from all axes,
        # except only 0:p for the specified for right, and -p: for left

        ind_right = [slice(-p, None) if i == ax else slice(None)
                     for i in range(ndim)]
        ind_left = [slice(0, p) if i == ax else slice(None)
                    for i in range(ndim)]
        if ax == 1:
            right = tf.roll(
                tensor[ind_right],
                shift=tensor.get_shape().as_list()[2]//2,
                axis=2)
            left = tf.roll(
                tensor[ind_left],
                shift=tensor.get_shape().as_list()[2]//2,
                axis=2)
        else:
            right = tensor[ind_right]
            left = tensor[ind_left]
        middle = tensor
        if one_sided:
            tensor = tf.concat([middle, left], axis=ax)
        else:
            tensor = tf.concat([right, middle, left], axis=ax)

    return tensor


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True):
    """2D Convolution-Batch Normalization-Activation stack builder.

    # Args:
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization

    # Returns:
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='valid',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if kernel_size == 3:
        periodic_padding_layer = Lambda(periodic_padding)
        x = periodic_padding_layer(x)

    x = conv(x)
    if batch_normalization:
        x = BatchNormalization()(x)
    if activation is not None:
        x = Activation(activation)(x)
    return x


def resnet(input_shape, depth, num_classes=16):
    """Stacks of 2 x (3 x 3) Conv2D-BN-ReLU.

    # Args:
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns:
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44)')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    model = Model(inputs=inputs, outputs=outputs)
    return model
