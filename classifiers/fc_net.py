import numpy as np

from layers.layers import *
from layers.layer_utils import *
from linear_classifier import softmax_loss

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """
    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        self.params['W1'] = np.random.normal(0, weight_scale, (input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim, dtype=np.float64)
        self.params['W2'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        self.params['b2'] = np.zeros(num_classes, dtype=np.float64)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']

        out1, cache1 = affine_relu_forward(X, W1, b1)
        scores, cache2 = affine_forward(out1, W2, b2)

        if y is None:
            return scores

        grads = {}

        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(np.square(W1)) + np.sum(np.square(W2)))

        dout1, dW2, db2 = affine_backward(dscores, cache2)
        dX, dW1, db1 = affine_relu_backward(dout1, cache1)
        dW1 += W1 * self.reg
        dW2 += W2 * self.reg

        grads['W1'], grads['W2'], grads['b1'], grads['b2'] = dW1, dW2, db1, db2

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.dtype = dtype
        self.num_layers = 1 + len(hidden_dims)
        self.params = {}

        for i in xrange(self.num_layers):
            indim = hidden_dims[i - 1] if i != 0 else input_dim
            outdim = hidden_dims[i] if i != self.num_layers - 1 else num_classes
            self.params['W' + str(i + 1)] = np.random.normal(0, weight_scale, (indim, outdim))
            self.params['b' + str(i + 1)] = np.zeros(outdim)

            if self.use_batchnorm and i != self.num_layers - 1:
                self.params['gamma' + str(i + 1)] = np.ones(outdim)
                self.params['beta' + str(i + 1)] = np.zeros(outdim)

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

        # dropout
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # batchnorm
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param
        # since they behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        # Forward
        loss = 0.0
        out = None
        caches = []
        for i in xrange(self.num_layers - 1):
            loss += 0.5 * self.reg * np.sum(np.square(self.params['W' + str(i + 1)]))

            input_data = out if i != 0 else X

            cache_bi, cache_di = None, None
            out, cache_ai = affine_forward(input_data, self.params['W' + str(i + 1)], self.params['b' + str(i + 1)])
            if self.use_batchnorm:
                out, cache_bi = batchnorm_forward(out, self.params['gamma' + str(i + 1)],
                                   self.params['beta' + str(i + 1)], self.bn_params[i])

            out, cache_ri = relu_forward(out)
            if self.use_dropout:
                out, cache_di = dropout_forward(out, self.dropout_param)

            caches.append((cache_ai, cache_bi, cache_ri, cache_di))

        loss += 0.5 * self.reg * np.sum(np.square(self.params['W' + str(self.num_layers)]))
        scores, cache_scores = affine_forward(out, self.params['W' + str(self.num_layers)], self.params['b' + str(self.num_layers)])

        # If test mode return early
        if mode == 'test':
            return scores

        loss_softmax, dscores = softmax_loss(scores, y)
        loss += loss_softmax

        # Backward
        grads = {}

        dout, grads['W' + str(self.num_layers)], grads['b' + str(self.num_layers)] = affine_backward(dscores, cache_scores)
        grads['W' + str(self.num_layers)] += self.reg * self.params['W' + str(self.num_layers)]

        for i in xrange(self.num_layers - 1):
            index = self.num_layers - 2 - i
            cache_ai, cache_bi, cache_ri, cache_di = caches[index]

            if self.use_dropout:
                dout = dropout_backward(dout, cache_di)

            dout = relu_backward(dout, cache_ri)
            if self.use_batchnorm:
                dout, grads['gamma' + str(index + 1)], grads['beta' + str(index + 1)] = batchnorm_backward(dout, cache_bi)

            dout, grads['W' + str(index + 1)], grads['b' + str(index + 1)] = affine_backward(dout, cache_ai)
            grads['W' + str(index + 1)] += self.reg * self.params['W' + str(index + 1)]

        return loss, grads