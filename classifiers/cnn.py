import numpy as np

from classifiers.linear_classifier import softmax_loss
from layers.layers import *
from layers.layer_utils import *
from layers.fast_conv_layers import *

class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """

        # Store weights and biases for the convolutional layer using the keys 'W1'
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases
        # of the output affine layer.
        C, H, W = input_dim

        self.reg = reg
        self.dtype = dtype
        self.params = {}

        self.params['W1'] = np.random.normal(0, weight_scale, (num_filters, C, filter_size, filter_size))
        self.params['b1'] = np.zeros(num_filters)

        self.params['W2'] = np.random.normal(0, weight_scale, (num_filters * H * W / 4, hidden_dim))
        self.params['b2'] = np.zeros(hidden_dim)

        self.params['W3'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        self.params['b3'] = np.zeros(num_classes)

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        c, conv_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        a1, affine_cache_1 = affine_relu_forward(c, W2, b2)
        scores, affine_cache_2 = affine_forward(a1, W3, b3)

        if y is None:
            return scores

        loss, grads = 0, {}

        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))

        da1, dW3, db3 = affine_backward(dscores, affine_cache_2)
        dc, dW2, db2 = affine_relu_backward(da1, affine_cache_1)
        _, dW1, db1 = conv_relu_pool_backward(dc, conv_cache)

        grads['W1'] = dW1 + W1 * self.reg
        grads['b1'] = db1
        grads['W2'] = dW2 + W2 * self.reg
        grads['b2'] = db2
        grads['W3'] = dW3 + W3 * self.reg
        grads['b3'] = db3

        return loss, grads

