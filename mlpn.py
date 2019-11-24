import numpy as np
from loglinear import softmax
from utils import *

STUDENT = {'name': 'YOUR NAME',
		   'ID': 'YOUR ID NUMBER'}


def get_weights_from_params_list(params):
	weights = []
	for i in range(len(params) - 2, -1, -2):
		weights.append(params[i])
	return weights


def classifier_output(x, params):
	out = x
	hidden_layer_inputs = []
	for W, b in zip(params[::2], params[1::2]):
		hidden_layer_inputs.append(out)
		hidden_layer = out.dot(W) + b
		out = np.tanh(hidden_layer)
	probs = softmax(hidden_layer), hidden_layer_inputs

	return probs


def predict(x, params):
	probs, _ = classifier_output(x, params)
	return np.argmax(probs)


def loss_and_gradients(x, y, params):
	"""
	params: a list as created by create_classifier(...)

	returns:
		loss,[gW1, gb1, gW2, gb2, ...]

	loss: scalar
	gW1: matrix, gradients of W1
	gb1: vector, gradients of b1
	gW2: matrix, gradients of W2
	gb2: vector, gradients of b2
	...

	(of course, if we request a linear classifier (ie, params is of length 2),
	you should not have gW2 and gb2.)
	"""
	y_pred, hidden_layer_inputs = classifier_output(x, params)
	y_vec = one_hot_vector(y, y_pred)
	# calculate loss
	loss = 0
	if y_pred[y] > 0:
		loss = -np.log(y_pred[y])

	grads = []
	# extract all the weights from params list (every even cell)
	weights = get_weights_from_params_list(params)
	weights_size = len(weights)
	b = hidden_layer_inputs[::-1]
	dz = y_pred - y_vec
	# flats the vector.
	dz = dz.reshape(-1, 1)
	for i in range(weights_size):
		if i != 0:
			dz = dz.T.dot((weights[i - 1]).T * derivative_tanh(b[i - 1])).T
		gb = dz
		gW = np.dot(dz, b[i].reshape(-1, 1).T)
		gW = gW.T
		grads.append(gb)
		grads.append(gW)

	grads = grads[::-1]

	return loss, grads


def create_classifier(dims):
	"""
	returns the parameters for a multi-layer perceptron with an arbitrary number
	of hidden layers.
	dims is a list of length at least 2, where the first item is the input
	dimension, the last item is the output dimension, and the ones in between
	are the hidden layers.
	For example, for:
		dims = [300, 20, 30, 40, 5]
	We will have input of 300 dimension, a hidden layer of 20 dimension, passed
	to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
	an output of 5 dimensions.

	Assume a tanh activation function between all the layers.

	return:
	a flat list of parameters where the first two elements are the W and b from input
	to first layer, then the second two are the matrix and vector from first to
	second layer, and so on.
	"""
	params = []
	for in_dim, out_dim in zip(dims, dims[1:]):
		W = np.random.randn(in_dim, out_dim) * np.sqrt(2 / (out_dim + in_dim))
		b = np.random.randn(out_dim) * np.sqrt(1 / (out_dim))
		b.reshape(b.shape[0], 1)
		params.append(W)
		params.append(b)
	return params
