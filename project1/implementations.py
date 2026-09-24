"""The six machine learning methods required for Project 1, Step 2."""

import numpy as np


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
	"""Train linear regression with gradient descent."""
	raise NotImplementedError


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
	"""Train linear regression with stochastic gradient descent."""
	raise NotImplementedError


def least_squares(y, tx):
	"""Train linear regression with the normal equations."""
	raise NotImplementedError


def ridge_regression(y, tx, lambda_):
	"""Train ridge regression with the normal equations."""
	raise NotImplementedError


def logistic_regression(y, tx, initial_w, max_iters, gamma):
	"""Train logistic regression with gradient descent."""
	raise NotImplementedError


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
	"""Train regularized logistic regression with gradient descent."""
	raise NotImplementedError
