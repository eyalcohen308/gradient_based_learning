import random
import utils as ut
import numpy as np
import mlp1 as mpl1

STUDENT = {'name': 'Eyal Cohen',
           'ID': '308347244'}


def test_predict(test_dataset, params):
	with open('test.pred', 'w') as file:
		for label, features in test_dataset:
			x = feats_to_vec(features)
			y_hat = mpl1.predict(x, params)
			file.write(ut.I2L[y_hat] + '\n')


def feats_to_vec(features):
	local_f2i = ut.F2I
	feat_vec = np.zeros(len(local_f2i))
	for feature in features:
		if feature in local_f2i:
			feat_vec[local_f2i[feature]] += 1
	# Should return a numpy vector of features.
	return feat_vec


def accuracy_on_dataset(dataset, params):
	good = bad = 0.0
	local_l2i = ut.L2I
	for label, features in dataset:
		feat_vec = feats_to_vec(features)
		y_hat = mpl1.predict(feat_vec, params)

		if local_l2i[label] == y_hat:
			good += 1
		else:
			bad += 1

	# Compute the accuracy (a scalar) of the current parameters
	# on the dataset.
	# accuracy is (correct_predictions / all_predictions)

	return good / (good + bad)


def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
	"""
	Create and train a classifier, and return the parameters.

	train_data: a list of (label, feature) pairs.
	dev_data  : a list of (label, feature) pairs.
	num_iterations: the maximal number of training iterations.
	learning_rate: the learning rate to use.
	params: list of parameters (initial values)
	"""
	for I in range(num_iterations):
		cum_loss = 0.0  # total loss in this iteration.
		random.shuffle(train_data)
		for label, features in train_data:
			x = feats_to_vec(features)  # convert features to a vector.
			y = ut.L2I[label]  # convert the label to number if needed.
			loss, grads = mpl1.loss_and_gradients(x, y, params)
			cum_loss += loss

			# update the parameters according to the gradients
			# and the learning rate.
			params[0] -= grads[0] * learning_rate
			params[1] -= grads[1] * learning_rate

		train_loss = cum_loss / len(train_data)
		train_accuracy = accuracy_on_dataset(train_data, params)
		dev_accuracy = accuracy_on_dataset(dev_data, params)
		print(I, train_loss, train_accuracy, dev_accuracy)
	return params


if __name__ == '__main__':
	# write code to load the train and dev sets, set up whatever you need,
	# and call train_classifier.

	train_data = ut.TRAIN
	dev_data = ut.DEV
	test_data = ut.TEST
	num_iterations = 100
	learning_rate = 0.001

	in_dim, out_dim = len(ut.F2I), len(ut.L2I)
	hid_dim = 100
	params = mpl1.create_classifier(in_dim, hid_dim, out_dim)
	trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)
	test_predict(test_data, trained_params)
