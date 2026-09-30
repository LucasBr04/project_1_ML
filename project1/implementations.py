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


def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    e = y - tx @ w                  
    return -tx.T @ e / len(y) 


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
	"""Train linear regression with gradient descent.

	Args:
		y: shape=(N, )
		tx: shape=(N, D)
		initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
		max_iters: a scalar denoting the total number of iterations of GD
		gamma: a scalar denoting the stepsize

	Returns:
		w: the last weight vector, shape=(D, )
		loss: the MSE (scalar) corresponding to w
	"""
	w = initial_w
	for n_iter in range(max_iters):
		grad = compute_gradient(y, tx, w)   # 1. sentir la pente
		w = w - gamma * grad                # 2. faire un pas vers le bas
	loss = mse(y, tx, w)
	return w, loss


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.

    Args:
        y: shape=(B, )
        tx: shape=(B, D)
        w: shape=(D, ). The vector of model parameters.

    Returns:
        An array of shape (D, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    e = y - tx @ w
    return -tx.T @ e / len(y)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
	"""Train linear regression with stochastic gradient descent (batch size 1).

	Args:
		y: shape=(N, )
		tx: shape=(N, D)
		initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
		max_iters: a scalar denoting the total number of iterations of SGD
		gamma: a scalar denoting the stepsize

	Returns:
		w: the last weight vector, shape=(D, )
		loss: the MSE (scalar) corresponding to w
	"""
	w = initial_w
	N = len(y)
	for n_iter in range(max_iters):
		i = np.random.randint(N)                                        # tirer 1 personne au hasard
		grad = compute_stoch_gradient(y[i:i+1], tx[i:i+1], w)   # pente d'1 personne
		w = w - gamma * grad                                            # un pas
	loss = mse(y, tx, w)
	return w, loss


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


def sigmoid(t):
	"""Apply the sigmoid function on t.

	Args:
		t: scalar or numpy array

	Returns:
		scalar or numpy array
	"""
	return 1.0 / (1.0 + np.exp(-t))


def logistic_loss(y, tx, w):
	"""Compute the cost by negative log likelihood.

	Args:
		y: shape=(N, ), labels in {0, 1}
		tx: shape=(N, D)
		w: shape=(D, )

	Returns:
		a non-negative loss (scalar)
	"""
	z = tx @ w
	# log(1 + exp(z)) computed in a numerically stable way
	return np.mean(np.logaddexp(0, z) - y * z)


def logistic_gradient(y, tx, w):
	"""Compute the gradient of the logistic loss.

	Args:
		y: shape=(N, ), labels in {0, 1}
		tx: shape=(N, D)
		w: shape=(D, )

	Returns:
		a vector of shape (D, )
	"""
	return tx.T @ (sigmoid(tx @ w) - y) / len(y)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
	"""Train logistic regression with gradient descent.

	Args:
		y: shape=(N, ), labels in {0, 1}
		tx: shape=(N, D)
		initial_w: shape=(D, ). The initial guess for the model parameters
		max_iters: a scalar denoting the total number of iterations of GD
		gamma: a scalar denoting the stepsize

	Returns:
		w: the last weight vector, shape=(D, )
		loss: the logistic loss (scalar) corresponding to w
	"""
	w = initial_w
	for n_iter in range(max_iters):
		grad = logistic_gradient(y, tx, w)
		w = w - gamma * grad
	loss = logistic_loss(y, tx, w)
	return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
	"""Train regularized logistic regression with gradient descent.

	Args:
		y: shape=(N, ), labels in {0, 1}
		tx: shape=(N, D)
		lambda_: scalar, the regularization parameter
		initial_w: shape=(D, ). The initial guess for the model parameters
		max_iters: a scalar denoting the total number of iterations of GD
		gamma: a scalar denoting the stepsize

	Returns:
		w: the last weight vector, shape=(D, )
		loss: the logistic loss (scalar) corresponding to w, without the penalty term
	"""
	w = initial_w
	for n_iter in range(max_iters):
		# gradient of lambda * ||w||^2 is 2 * lambda * w
		grad = logistic_gradient(y, tx, w) + 2 * lambda_ * w
		w = w - gamma * grad
	loss = logistic_loss(y, tx, w)
	return w, loss
