"""The six machine learning methods required for Project 1, Step 2."""

import numpy as np

def mse(y, tx, w):
    """Calculate the loss using MSE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    N = tx.shape[0]
    C = 1/(2*N)
    return C * np.sum((y-tx@w)**2)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
	"""Train linear regression with gradient descent."""
	raise NotImplementedError


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
	"""Train linear regression with stochastic gradient descent."""
	raise NotImplementedError



def least_squares(y, tx):
	"""Calculate the least squares solution.
	returns mse, and optimal weights.

	Args:
	y: numpy array of shape (N,), N is the number of samples.
	tx: numpy array of shape (N,D), D is the number of features.

	Returns:
	w: optimal weights, numpy array of shape(D,), D is the number of features.
	mse: scalar.
	"""
	tx_t = np.transpose(tx)
	A = (tx_t @ tx)
	b = tx_t @ y
	w = np.linalg.solve(A, b)
	mse_res = mse(y, tx, w)
	return w, mse_res


def ridge_regression(y, tx, lambda_):
	"""implement ridge regression.

	Args:
		y: numpy array of shape (N,), N is the number of samples.
		tx: numpy array of shape (N,D), D is the number of features.
		lambda_: scalar.

	Return:
		w: optimal weights, numpy array of shape(D,), D is the number of features.
	"""
	tx_t = np.transpose(tx)
	N = tx.shape[0]
	A = tx_t @ tx + lambda_*2*N*np.identity(tx.shape[1])
	b = tx_t @ y
	w = np.linalg.solve(A, b)
	mse_res = mse(y, tx, w)
	return w, mse_res


def logistic_regression(y, tx, initial_w, max_iters, gamma):
	"""Train logistic regression with gradient descent."""
	raise NotImplementedError


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
	"""Train regularized logistic regression with gradient descent."""
	raise NotImplementedError
