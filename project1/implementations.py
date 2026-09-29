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
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    """
	# Define parameters to store w and loss
	ws = [initial_w]
	losses = []
	w = initial_w
	for n_iter in range(max_iters):
		grad = compute_gradient(y, tx, w)   # 1. sentir la pente
		loss = mse(y, tx, w)       # (pour suivre l'altitude)
		w = w - gamma * grad                # 2. faire un pas vers le bas
		# store w and loss
		ws.append(w)
		losses.append(loss)
		print(
			"GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
				bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
			)
		)
	return losses, ws


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    e = y - tx @ w
    return -tx.T @ e / len(y)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
	"""The Stochastic Gradient Descent algorithm (SGD).
    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """
	ws = [initial_w]
	losses = []
	w = initial_w
	for n_iter in range(max_iters):
		for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
			grad = compute_stoch_gradient(minibatch_y, minibatch_tx, w)  # pente d'1 personne
			w = w - gamma * grad                                         # un pas
			loss = mse(y, tx, w)   # seulement pour afficher le progrès
			ws.append(w)
			losses.append(loss)
		print(
            "SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )
		return losses, ws


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
