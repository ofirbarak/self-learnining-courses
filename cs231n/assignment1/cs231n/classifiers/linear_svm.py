import numpy as np
from random import shuffle

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
    dW = np.zeros(W.shape) # initialize the gradient as zero
    
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        sum_ones = 0
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                sum_ones += 1
                
            # update dW
            dW[:, j] += (margin > 0)*X[i]
#         dW[:, y[i]] += -1*sum_ones*X[i]
        
                

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################


    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    num_examples = X.shape[0]
    
    scores = np.matmul(X, W)
    
    # Subtract the correct label from each example
    true_labels_score = [scores[i, y[i]] for i in range(X.shape[0])]
    margin = np.array(true_labels_score)
    difference = (scores.T - margin.T).T + 1
    difference[difference < 0] = 0
    
    # Set the value in positions of the true labels in the difference matrix to zero
    true_labels = np.arange(X.shape[0])
    difference[true_labels, y[true_labels]] = 0
    
    # Sum the error, divide the number of examples and multiply by the regularization term
    loss = np.sum(difference)
    loss /= num_examples
    loss += reg * np.sum(W * W)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    C = W.shape[1]
    D = W.shape[0]
    N = X.shape[0]
    
    diff = (difference > 0) * 
    A = np.repeat(difference, np.repeat(D, C), axis=1).reshape(N,D,C)
    A = np.sum(A, axis=0)
    dW += A
    
    # y_ update
    B = np.zeros((N,D,C))
    
    
    print(A.shape)
#     sum_ones = np.sum(difference, axis=1)
#     xs += -1*np.dot(sum_ones, X.T).T

    dW /= N
    dW += 2 * reg * W
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
