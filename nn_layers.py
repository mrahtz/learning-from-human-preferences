import tensorflow as tf

"""
Wrappers for TensorFlow's layers integrating batchnorm in the right place.
"""


def conv_layer(x, filters, kernel_size, strides, batchnorm, training, name,
               reuse, activation='relu'):
    x = tf.layers.conv2d(
        x,
        filters,
        kernel_size,
        strides,
        activation=None,
        name=name,
        reuse=reuse)

    if batchnorm:
        batchnorm_name = name + "_batchnorm"
        x = tf.layers.batch_normalization(
            x, training=training, reuse=reuse, name=batchnorm_name)

    if activation == 'relu':
        x = tf.nn.leaky_relu(x, alpha=0.01)
    else:
        raise Exception("Unknown activation for conv_layer", activation)

    return x


def dense_layer(x,
                units,
                name,
                reuse,
                activation=None):
    x = tf.layers.dense(x, units, activation=None, name=name, reuse=reuse)

    if activation is None:
        pass
    elif activation == 'relu':
        x = tf.nn.leaky_relu(x, alpha=0.01)
    else:
        raise Exception("Unknown activation for dense_layer", activation)

    return x