import numpy as np

"""
    CALL svm_loss AND softmax_loss FOR MODULAR!
"""

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)
    loss = 0.0
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in xrange(num_train):
        scores = X[i, :].dot(W)
        correct_class_score = scores[y[i]]
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i, :]
                dW[:, y[i]] -= X[i, :]

    loss /= num_train
    dW /= num_train

    loss += reg * np.sum(np.square(W))
    dW += 2 * reg * W

    return loss, dW

def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    N, D = X.shape
    scores = X.dot(W)
    scores = scores - scores[np.arange(N), y].reshape(-1, 1) + 1
    scores2 = np.maximum(scores, 0)
    scores2[np.arange(N), y] = 0
    loss = np.sum(scores2) * 1.0 / N
    loss += reg * np.sum(np.square(W))

    dscores = np.ones_like(scores, dtype=loss.dtype) / N
    dscores[scores < 0] = 0
    dscores[np.arange(N), y] += -1 * np.sum(dscores, 1)
    dW = X.T.dot(dscores)
    dW += 2 * reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    N, D = X.shape
    scores = X.dot(W)
    loss = np.sum(-1 * scores[np.arange(N), y]) + np.sum(np.log(np.sum(np.exp(scores), 1)))
    loss /= N
    loss += reg * np.sum(np.square(W))

    scores_e = np.exp(scores)
    dscore = scores_e / np.sum(scores_e, 1).reshape(N, 1)
    dscore[np.arange(N), y] = dscore[np.arange(N), y] - 1
    dscore /= N
    dW = X.T.dot(dscore)
    dW += 2 * reg * W

    return loss, dW


# another kind of loss compution
def svm_loss(scores, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - scores: Input data, of shape (N, C) where scores[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dscores: Gradient of the loss with respect to dscores
    """
    N = scores.shape[0]
    scores = scores - scores[np.arange(N), y].reshape(-1, 1) + 1
    scores2 = np.maximum(scores, 0)
    scores2[np.arange(N), y] = 0
    loss = np.sum(scores2) * 1.0 / N

    dscores = np.ones_like(scores, dtype=loss.dtype) / N
    dscores[scores < 0] = 0
    dscores[np.arange(N), y] += -1 * np.sum(dscores, 1)

    return loss, dscores


def softmax_loss(scores, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - scores: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dscores: Gradient of the loss with respect to x
    """
    N, C = scores.shape
    scores = scores - np.max(scores, 1, keepdims=True)
    loss = np.sum(-1 * scores[np.arange(N), y]) + np.sum(np.log(np.sum(np.exp(scores), 1)))
    loss /= N

    scores_e = np.exp(scores)
    dscores = scores_e / np.sum(scores_e, 1).reshape(N, 1)
    dscores[np.arange(N), y] = dscores[np.arange(N), y] - 1

    dscores /= N

    return loss, dscores


class LinearClassifier(object):

    def __init__(self):
        self.W = None


    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_train, dim = X.shape
        num_class = np.max(y) + 1
        if self.W == None:
            # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_class)

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in xrange(num_iters):
            ids = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[ids, :]
            y_batch = y[ids]

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)
            self.W += -learning_rate * grad

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history


    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        y_pred = np.argmax(X.dot(self.W), axis=1)
        return y_pred


    def loss(self, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """
        pass


class LinearSVM(LinearClassifier):
    """ A subclass that uses the Multiclass SVM loss function """
    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """ A subclass that uses the Softmax """
    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)








