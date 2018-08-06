import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_examples = X.shape[0]
    dim = X.shape[1]
    num_classes = W.shape[1]
    
    for i in range(num_examples):
        f = X[i].dot(W)
        exp_score = np.exp(f)
        one_example_loss = f - np.max(f)
        one_example_loss = np.exp(one_example_loss) / np.sum(np.exp(one_example_loss))
        loss += -np.log(one_example_loss[y[i]])
        
        exp_sum = np.sum(np.exp(f))
        for j in range(num_classes):
            one_dw = np.ones((dim)) * (exp_score[j] / exp_sum)
            one_dw += -1 * (j == y[i])
            dW[:,j] += one_dw*X[i]
        
     # Normalize
    loss /= float(num_examples)
    dW /= float(num_examples)
    
    # Adding regularization term
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * np.sum(W * W)
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_examples = X.shape[0]
    dim = X.shape[1]
    num_classes = W.shape[1]
    
    scores = X.dot(W)
    exp_scores = np.exp(scores)
    
    max_score_like = np.max(scores, axis=1)[:,np.newaxis].dot(np.ones((1,num_classes)))
    f = scores - max_score_like
    softmax_score = (np.exp(f).T / np.sum(np.exp(f), axis=1)).T
    
    softmax_der = np.array(softmax_score)
    indexes = np.arange(num_examples)
    softmax_der[indexes, y[indexes]] -= 1
    
    dW = X.T.dot(softmax_der)
    loss = np.sum(-np.log(softmax_score[indexes, y[indexes]]))
    
    loss /= float(num_examples)
    dW /= float(num_examples)
    
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

