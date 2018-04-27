import tensorflow as tf


class ModelConfig(object):
    """
    Model build configs. Options:
    nconv: int (default 2): number of convolutional layers
    nfilters: int or int[] (default [32, 64]): number of filters, use list to specify
        different number of filters for each layer
    stride: int[4] (default [1,1,1,1]): filter stride
    kernel: int or int[]: filter size (list: at each layer)
    maxpool: int or int[]: maxpool sample count (list: at each layer)
    dropout: float or float[]: dropout rate (list: at each layer)
    fc_units: int: Number of neurons in the fully connected layer
    """

    def __init__(self, **kwargs):
        self.nconv = kwargs.get('nconv', 2)
        self.nfilters = kwargs.get('nfilters', [32, 64])
        self.stride = kwargs.get('stride', [1, 1, 1, 1])
        self.kernel_size = kwargs.get('kernel', 5)
        self.maxpool = kwargs.get('maxpool', 2)
        self.dropout = kwargs.get('dropout', 0.2)
        self.fc_units = kwargs.get('fc_units', 1024)

    def get(self, attr, layer=None):
        if not hasattr(self, attr):
            return None

        val = getattr(self, attr)
        if layer is not None and isinstance(val, list):
            if layer < len(val):
                return val[layer]
            else:
                return val[-1]

        return val


def conv_layer(in_tensors, configs, layer):
    kernel_size = configs.get('kernel', layer)
    nfilters = configs.get('nfilters', layer)
    stride = configs.get('stride')

    w = tf.get_variable(
        'weights',
        [kernel_size, kernel_size, in_tensors.get_shape()[3], nfilters],
        tf.float32,
        tf.contrib.layers.xavier_initializer()
    )
    b = tf.get_variable(
        'b',
        [nfilters, ],
        tf.float32,
        tf.constant_initializer(0.0)
    )
    return tf.nn.leaky_relu(tf.nn.conv2d(in_tensors, w, stride, 'SAME') + b)


def maxpool_layer(in_tensors, configs, layer):
    samples = configs.get('maxpool', layer)
    return tf.nn.max_pool(
        in_tensors,
        [1, samples, samples, 1],
        [1, samples, samples, 1],
        'SAME'
    )


def dropout(in_tensors, is_training, configs, layer):
    keep = 1.0 - configs.get('dropout', layer)
    return tf.cond(is_training, lambda: tf.nn.dropout(in_tensors, keep), lambda: in_tensors)


def fc_layer(in_tensors, configs, layer):
    return tf.nn.leaky_relu(fc_no_activation_layer(in_tensors, configs.get('fc_units')))


def fc_no_activation_layer(in_tensors, configs, layer):
    n_units = configs.get('fc_units')
    w = tf.get_variable('fc_W',
                        [in_tensors.get_shape()[1], n_units],
                        tf.float32,
                        tf.contrib.layers.xavier_initializer())
    b = tf.get_variable('fc_B',
                        [n_units, ],
                        tf.float32,
                        tf.constant_initializer(0.0))
    return tf.matmul(in_tensors, w) + b


def build_model(in_tensors, configs, is_training):
    layer = 0
    out_layers = [in_tensors]

    # Convolutional layers with maxpooling and dropout
    for i in range(1, configs.get('nconv') + 1):
        with tf.variable_scope('conv{}'.format(i)):
            # l1 = maxpool_layer(conv_layer(in_tensors, 5, 32), 2)
            partial = maxpool_layer(conv_layer(out_layers[-1], configs, layer), configs, layer)
            out_layers.append(dropout(partial, is_training, configs, layer))
        layer += 1

    with tf.variable_scope('flatten'):
        out_layers.append(tf.layers.flatten(out_layers[-1]))

    # Fully collected layer, 1024 neurons, 40% dropout
    with tf.variable_scope('fc'):
        l3 = fc_layer(out_layers[-1], configs)
        out_layers.append(dropout(l3, is_training, configs, layer))
    layer += 1

    # Output
    with tf.variable_scope('out'):
        out_layers.append(fc_no_activation_layer(out_layers[-1], configs, layer))
    layer += 1

    return out_layers[-1]
