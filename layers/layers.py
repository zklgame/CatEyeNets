import numpy as np

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    cache = (x, w, b)
    x = x.reshape(x.shape[0], -1)
    out = x.dot(w) + b

    return out, cache

def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx = dout.dot(w.T)
    dw = x.reshape(dx.shape).T.dot(dout)
    dx = dx.reshape(x.shape)
    db = np.sum(dout, 0)

    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = np.maximum(x, 0)
    cache = x

    return out, cache

def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    x = cache
    dx = dout
    dx[x < 0] = 0

    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    cache = None
    if mode == 'train':
        sample_mean = np.mean(x, 0)
        sample_var = np.var(x, 0)
        x_normalized = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = gamma * x_normalized + beta

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        cache = (x, gamma, beta, eps)

        bn_param['running_mean'] = running_mean
        bn_param['running_var'] = running_var

    elif mode == 'test':
        x = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x + beta

    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    return out, cache

def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    x, gamma, beta, eps = cache
    N = x.shape[0]
    sample_mean = np.mean(x, 0)
    sample_var = np.var(x, 0)
    x_normalized = (x - sample_mean) / np.sqrt(sample_var + eps)

    dgamma = np.sum(dout * x_normalized, 0)
    dbeta = np.sum(dout, 0)

    dx_normalized = dout * gamma

    dx_norm_numerator = dx_normalized / np.sqrt(sample_var + eps)
    dx = dx_norm_numerator
    dsample_mean = np.sum(-1 * dx_norm_numerator, 0)
    dx += dsample_mean / N

    dx_norm_denominator_2 = -0.5 * np.sum(dx_normalized * (x - sample_mean), 0) / np.power(sample_var + eps, 1.5)
    dsample_var = dx_norm_denominator_2
    dx += 2 * (x - sample_mean) / N * dsample_var
    dsample_mean = -1 * 2 * np.sum(x - sample_mean, 0) / N * dsample_var
    dx += dsample_mean / N

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    if mode == 'train':
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
    elif mode == 'test':
        out = x
    else:
        raise ValueError('Invalid dropout mode "%s"' % mode)

    cache = mask

    return out, cache

def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    Outputs:
    - dx: Gradient with respect to inputs x
    """
    mask = cache

    dx = dout * mask

    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    S = conv_param['stride']
    P = conv_param['pad']

    H_new = 1 + (H + 2 * P - HH) / S
    W_new = 1 + (W + 2 * P - WW) / S

    out = np.zeros((N, F, H_new, W_new), dtype=w.dtype)

    for i in xrange(N):
        xi = x[i, :, :, :]
        xi_pad = np.pad(xi, ((0,), (P,), (P,)), 'constant',  constant_values=(0,))

        for j in xrange(F):
            wj = w[j, :, :, :]
            for hh in xrange(H_new):
                for ww in xrange(W_new):
                    out[i, j, hh, ww] = np.sum(xi_pad[:, hh * S : hh * S + HH, ww * S : ww * S + WW] * wj) + b[j]

    cache = (x, w, b, conv_param)

    return out, cache

def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    S = conv_param['stride']
    P = conv_param['pad']

    H_new = 1 + (H + 2 * P - HH) / S
    W_new = 1 + (W + 2 * P - WW) / S

    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    for i in xrange(N):
        xi = x[i, :, :, :]
        xi_pad = np.pad(xi, ((0,), (P,), (P,)), 'constant',  constant_values=(0,))
        dxi_pad = np.zeros_like(xi_pad)

        for j in xrange(F):
            wj = w[j, :, :, :]
            for hh in xrange(H_new):
                for ww in xrange(W_new):
                    db[j] += dout[i, j, hh, ww]
                    dw[j, :, :, :] += dout[i, j, hh, ww] * xi_pad[:, hh * S : hh * S + HH, ww * S : ww * S + WW]
                    dxi_pad[:, hh * S : hh * S + HH, ww * S : ww * S + WW] +=  dout[i, j, hh, ww] * wj

        dx[i, :, :, :] += dxi_pad[:, P:P+H, P:P+W]

    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    N, C, H, W = x.shape
    HH, WW, S = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

    H_new = 1 + (H - HH) / S
    W_new = 1 + (W - WW) / S
    out = np.zeros((N, C, H_new, W_new), dtype=x.dtype)

    for n in xrange(N):
        for c in xrange(C):
            for hh in xrange(H_new):
                for ww in xrange(W_new):
                    out[n, c, hh, ww] = np.max(x[n, c, hh * S : hh * S + HH, ww * S : ww * S + WW])

    cache = (x, pool_param)

    return out, cache

def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    x, pool_param = cache
    N, C, H, W = x.shape
    HH, WW, S = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

    H_new = 1 + (H - HH) / S
    W_new = 1 + (W - WW) / S

    dx = np.zeros_like(x)

    for n in xrange(N):
        for c in xrange(C):
            for hh in xrange(H_new):
                for ww in xrange(W_new):
                    max_num = np.max(x[n, c, hh * S : hh * S + HH, ww * S : ww * S + WW])
                    mask = x[n, c, hh * S : hh * S + HH, ww * S : ww * S + WW] == max_num
                    dx[n, c, hh * S : hh * S + HH, ww * S : ww * S + WW][mask] += dout[n, c, hh, ww]

    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    N, C, H, W = x.shape
    x_new = x.transpose(0, 2, 3, 1).reshape(N * H * W, C)
    out, cache = batchnorm_forward(x_new, gamma, beta, bn_param)
    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    return out, cache

def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    N, C, H, W = dout.shape
    dout_new = dout.transpose(0, 2, 3, 1).reshape(N * H * W, C)
    dx, dgamma, dbeta = batchnorm_backward(dout_new, cache)
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    return dx, dgamma, dbeta





