import numpy as np
from loglinear import softmax
import utils as ut

STUDENT = {'name': 'YOUR NAME',
           'ID': 'YOUR ID NUMBER'}


def classifier_output(x, params):
	U, W, b, b_tag = params
	# compute the softmax with the given parameters
	h_layer_tanh = np.tanh(np.dot(x, W) + b)
	probs = softmax(np.dot(h_layer_tanh, U) + b_tag)
	return probs


def derivative_tanh(x):
	# return the derivative of tanh
	return 1 - (np.tanh(x) ** 2)


def predict(x, params):
	"""
	params: a list of the form [W, b, U, b_tag]
	"""
	return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
	"""
	params: a list of the form [W, b, U, b_tag]

	returns:
		loss,[gW, gb, gU, gb_tag]

	loss: scalar
	gW: matrix, gradients of W
	gb: vector, gradients of b
	gU: matrix, gradients of U
	gb_tag: vector, gradients of b_tag
	"""
	U, W, b, b_tag = params
	x = np.array(x)
	loss = 0
	# loss function:
	y_pred = classifier_output(x, params)
	y_vec = ut.one_hot_vector(y, y_pred)

	# Derivatives:
	gb_tag = y_pred - y_vec

	flat_gb_tag = gb_tag.reshape(-1, 1)
	h = np.dot(x, W) + b
	h_tanh = np.tanh(h)
	gU = (h_tanh * flat_gb_tag).T

	flat_gb_tag_T = flat_gb_tag.T
	dh_dz1 = U.T * derivative_tanh(h)
	flat_gb_tag_T_times_dh_dz1 = np.dot(flat_gb_tag_T, dh_dz1)
	gW = flat_gb_tag_T_times_dh_dz1.T.dot(x.reshape(-1, 1).T).T

	gb = flat_gb_tag_T_times_dh_dz1[0]

	if y_pred[y] > 0:
		# log of number under 0 is not defined.
		loss = -np.log(y_pred[y])

	return loss, [gU, gW, gb, gb_tag]


def create_classifier(in_dim, hid_dim, out_dim):
	"""
	returns the parameters for a multi-layer perceptron,
	with input dimension in_dim, hidden dimension hid_dim,
	and output dimension out_dim.

	return:
	a flat list of 4 elements, W, b, U, b_tag.
	"""
	# zeros init:
	# W = np.zeros((in_dim, hid_dim))
	# b = np.zeros(hid_dim)
	# b_tag = np.zeros(out_dim)
	# U = np.zeros((hid_dim, out_dim))

	# normalized random values init:
	W = np.random.randn(in_dim, hid_dim) * np.sqrt(2 / (hid_dim + in_dim))
	b = np.random.randn(hid_dim) * np.sqrt(1 / hid_dim)
	U = np.random.randn(hid_dim, out_dim) * np.sqrt(2 / (hid_dim + out_dim))
	b_tag = np.random.randn(out_dim) * np.sqrt(1 / out_dim)
	params = [U, W, b, b_tag]
	return params


if __name__ == '__main__':
	from grad_check import gradient_check

	U, W, b, b_tag = create_classifier(2, 2, 2)


	def _loss_and_W_grad(W):
		global U, b, b_tag
		loss, grads = loss_and_gradients([1, 2], 0, [U, W, b, b_tag])
		return loss, grads[1]


	def _loss_and_b_grad(b):
		global U, W, b_tag
		loss, grads = loss_and_gradients([1, 2], 0, [U, W, b, b_tag])
		return loss, grads[2]


	def _loss_and_U_grad(U):
		global W, b, b_tag
		loss, grads = loss_and_gradients([1, 2], 0, [U, W, b, b_tag])
		return loss, grads[0]


	def _loss_and_b_tag_grad(b_tag):
		global U, W, b
		loss, grads = loss_and_gradients([1, 2], 0, [U, W, b, b_tag])
		return loss, grads[3]


	for _ in range(10):
		W = np.random.randn(W.shape[0], W.shape[1])
		b = np.random.randn(b.shape[0])
		b_tag = np.random.randn(b_tag.shape[0])
		U = np.random.randn(U.shape[0], U.shape[1])
		gradient_check(_loss_and_W_grad, W)
		gradient_check(_loss_and_b_tag_grad, b_tag)
		gradient_check(_loss_and_b_grad, b)
		gradient_check(_loss_and_U_grad, U)
