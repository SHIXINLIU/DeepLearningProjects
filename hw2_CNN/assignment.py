from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data
from convolution import conv2d

import os
import tensorflow as tf
import numpy as np
import random

class Model(tf.keras.Model):
	def __init__(self):
		"""
		This model class will contain the architecture for your CNN that classifies images.
		Do not modify the constructor, as doing so will break the autograder. We have left
		in variables in the constructor for you to fill out, but you are welcome to change them if you'd like.
		"""
		super(Model, self).__init__()

		self.batch_size = 200
		self.num_classes = 2
		# TODO: Initialize all hyperparameters
		self.learnig_rate = 5e-4
		self.optimizer = tf.keras.optimizers.Adam(self.learnig_rate)

		self.epoch = 10

		self.layer1_filter_num = 32
		self.layer1_stride_size = 1
		self.layer1_pool_ksize = 2
		self.layer1_pool_stride = 2

		self.layer2_filter_num = 128
		self.layer2_stride_size = 1
		self.layer2_pool_ksize = 2
		self.layer2_pool_stride = 2

		self.layer3_filter_num = 200
		self.layer3_stride_size = 1
		self.layer3_pool_ksize = 2
		self.layer3_pool_stride = 2

		self.layer4_filter_num = 256
		self.layer4_stride_size = 1
		self.layer4_pool_ksize = 2
		self.layer4_pool_stride = 2

		self.flatten_width = 1024        # (32 / strides / 2^num_of_pooling)^2 * last_layer_filter_num
		self.dense1_output_width = 64
		self.dense2_output_width = 32
		self.dense3_output_width = 16

		# TODO: Initialize all trainable parameters
		self.filter1 = tf.Variable(tf.random.truncated_normal([3, 3, 3, self.layer1_filter_num], stddev=0.1))
		self.stride1 = [1, self.layer1_stride_size, self.layer1_stride_size, 1]
		self.filter2 = tf.Variable(tf.random.truncated_normal([3, 3, self.layer1_filter_num, self.layer2_filter_num], stddev=0.1))
		self.stride2 = [1, self.layer2_stride_size, self.layer2_stride_size, 1]
		self.filter3 = tf.Variable(tf.random.truncated_normal([3, 3, self.layer2_filter_num, self.layer3_filter_num], stddev=0.1))
		self.stride3 = [1, self.layer3_stride_size, self.layer3_stride_size, 1]
		self.filter4 = tf.Variable(	tf.random.truncated_normal([3, 3, self.layer3_filter_num, self.layer4_filter_num], stddev=0.1))
		self.stride4 = [1, self.layer4_stride_size, self.layer4_stride_size, 1]

		self.w1 = tf.Variable(tf.random.normal([self.flatten_width, self.dense1_output_width], stddev=.1, dtype=tf.float32))
		self.w2 = tf.Variable(tf.random.normal([self.dense1_output_width, self.dense2_output_width], stddev=.1, dtype=tf.float32))
		self.w3 = tf.Variable(tf.random.normal([self.dense2_output_width, self.dense3_output_width], stddev=.1, dtype=tf.float32))
		self.w4 = tf.Variable(tf.random.normal([self.dense3_output_width, self.num_classes], stddev=.1, dtype=tf.float32))
		self.b1 = tf.Variable(tf.random.normal([1, self.dense1_output_width], stddev=.1, dtype=tf.float32))
		self.b2 = tf.Variable(tf.random.normal([1, self.dense2_output_width], stddev=.1, dtype=tf.float32))
		self.b3 = tf.Variable(tf.random.normal([1, self.dense3_output_width], stddev=.1, dtype=tf.float32))
		self.b4 = tf.Variable(tf.random.normal([1, self.num_classes], stddev=.1, dtype=tf.float32))

	def call(self, inputs, is_testing=False):
		"""
		Runs a forward pass on an input batch of images.
		:param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
		:param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment
							and this function is being called during testing
		:return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
		"""
		# Remember that
		# shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)
		# shape of filter = (filter_height, filter_width, in_channels, out_channels)
		# shape of strides = (batch_stride, height_stride, width_stride, channels_stride)

		# layer1
		layer1_conv = tf.nn.conv2d(inputs, self.filter1, self.stride1, 'SAME')
		mean1, variance1 = tf.nn.moments(layer1_conv, axes=[0, 1, 2])
		layer1_norm = tf.nn.batch_normalization(layer1_conv, mean1, variance1, offset=None, scale=None, variance_epsilon=1e-3)
		layer1_relu = tf.nn.relu(layer1_norm)
		layer1_pool = tf.nn.max_pool(layer1_relu, self.layer1_pool_ksize, self.layer1_pool_stride, 'SAME')

		# layer2
		layer2_conv = tf.nn.conv2d(layer1_pool, self.filter2, self.stride2, 'SAME')
		mean2, variance2 = tf.nn.moments(layer2_conv, axes=[0, 1, 2])
		layer2_norm = tf.nn.batch_normalization(layer2_conv, mean2, variance2, offset=None, scale=None, variance_epsilon=1e-3)
		layer2_relu = tf.nn.relu(layer2_norm)
		layer2_pool = tf.nn.max_pool(layer2_relu, self.layer2_pool_ksize, self.layer2_pool_stride, 'SAME')

		# layer3
		layer3_conv = tf.nn.conv2d(layer2_pool, self.filter3, self.stride3, 'SAME')
		mean3, variance3 = tf.nn.moments(layer3_conv, axes=[0, 1, 2])
		layer3_norm = tf.nn.batch_normalization(layer3_conv, mean3, variance3, offset=None, scale=None, variance_epsilon=1e-3)
		layer3_relu = tf.nn.relu(layer3_norm)
		layer3_pool = tf.nn.max_pool(layer3_relu, self.layer3_pool_ksize, self.layer3_pool_stride, 'SAME')

		# layer4
		if is_testing is True:
			layer4_conv = conv2d(layer3_pool, self.filter4, self.stride4, 'SAME')
		else:
			layer4_conv = tf.nn.conv2d(layer3_pool, self.filter4, self.stride4, 'SAME')
		mean4, variance4 = tf.nn.moments(layer4_conv, axes=[0, 1, 2])
		layer4_norm = tf.nn.batch_normalization(layer4_conv, mean4, variance4, offset=None, scale=None, variance_epsilon=1e-3)
		layer4_relu = tf.nn.relu(layer4_norm)
		layer4_pool = tf.nn.max_pool(layer4_relu, self.layer4_pool_ksize, self.layer4_pool_stride, 'SAME')

		# flatten
		dense_input = tf.reshape(layer4_pool, [-1, self.flatten_width])

		# fully connected layer1
		dense_layer1 = tf.nn.relu(tf.matmul(dense_input, self.w1) + self.b1)
		dense_layer1 = tf.nn.dropout(dense_layer1, rate=0.3)

		# fully connected layer2
		dense_layer2 = tf.nn.relu(tf.matmul(dense_layer1, self.w2) + self.b2)
		dense_layer2 = tf.nn.dropout(dense_layer2, rate=0.3)

		# fully connected layer3
		dense_layer3 = tf.nn.relu(tf.matmul(dense_layer2, self.w3) + self.b3)
		dense_layer3 = tf.nn.dropout(dense_layer3, rate=0.3)

		# fully connected layer4
		dense_layer4 = tf.nn.relu(tf.matmul(dense_layer3, self.w4) + self.b4)

		logits = dense_layer4
		return logits


	def loss(self, logits, labels):
		"""
		Calculates the model cross-entropy loss after one forward pass.
		:param logits: during training, a matrix of shape (batch_size, self.num_classes) 
		containing the result of multiple convolution and feed forward layers
		Softmax is applied in this function.
		:param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
		:return: the loss of the model as a Tensor
		"""
		return tf.nn.softmax_cross_entropy_with_logits(labels, logits)


	def accuracy(self, logits, labels):
		"""
		Calculates the model's prediction accuracy by comparing
		logits to correct labels – no need to modify this.
		:param logits: a matrix of size (num_inputs, self.num_classes);
						during training, this will be (batch_size, self.num_classes)
						containing the result of multiple convolution and feed forward layers
		:param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training,
						this will be (batch_size, self.num_classes)

		NOTE: DO NOT EDIT
		
		:return: the accuracy of the model as a Tensor
		"""
		correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
		return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


def train(model, train_inputs, train_labels):
	'''
	Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs 
	and labels - ensure that they are shuffled in the same order using tf.gather.
	To increase accuracy, you may want to use tf.image.random_flip_left_right on your
	inputs before doing the forward pass. You should batch your inputs.
	:param model: the initialized model to use for the forward pass and backward pass
	:param train_inputs: train inputs (all inputs to use for training), shape (num_inputs, width, height, num_channels)
	:param train_labels: train labels (all labels to use for training), shape (num_labels, num_classes)
	:return: None
	'''
	# shuffle
	indices = tf.range(0, train_inputs.shape[0])
	indices = tf.random.shuffle(indices)
	train_inputs = tf.gather(train_inputs, indices)
	train_labels = tf.gather(train_labels, indices)

	for batch in range(int(train_inputs.shape[0]/model.batch_size)):
		start = batch * model.batch_size
		end = (batch + 1) * model.batch_size
		if (batch + 1) * model.batch_size > train_inputs.shape[0]:  # if the batch is out of range
			end = train_inputs.shape[0]
		inputs = tf.image.random_flip_left_right(train_inputs[start: end])  # random flip left right
		labels = train_labels[start: end]

		with tf.GradientTape() as tape:
			logits = model.call(inputs)
			loss = model.loss(logits, labels)

			if batch % 10 == 0:  # print training accuracy every 10 batches
				train_acc = model.accuracy(logits, labels)
				print("Accuracy on training set after {} images: {}".format(model.batch_size * batch, train_acc))

		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_inputs, test_labels):
	"""
	Tests the model on the test inputs and labels. You should NOT randomly flip images or do any extra preprocessing.
	:param test_inputs: test data (all images to be tested), shape (num_inputs, width, height, num_channels)
	:param test_labels: test labels (all corresponding labels), shape (num_labels, num_classes)
	:return: test accuracy - this can be the average accuracy across all batches
								or the sum as long as you eventually divide it by batch_size
	"""
	test_logits = model.call(test_inputs, is_testing=True)
	test_accuracy = model.accuracy(test_logits, test_labels)
	return test_accuracy


def visualize_results(image_inputs, probabilities, image_labels, first_label, second_label):
	"""
	Uses Matplotlib to visualize the results of our model.
	:param image_inputs: image data from get_data(), limited to 10 images, shape (10, 32, 32, 3)
	:param probabilities: the output of model.call(), shape (10, num_classes)
	:param image_labels: the labels from get_data(), shape (10, num_classes)
	:param first_label: the name of the first class, "dog"
	:param second_label: the name of the second class, "cat"

	NOTE: DO NOT EDIT

	:return: doesn't return anything, a plot should pop-up 
	"""
	predicted_labels = np.argmax(probabilities, axis=1)
	num_images = image_inputs.shape[0]

	fig, axs = plt.subplots(ncols=num_images)
	fig.suptitle("PL = Predicted Label\nAL = Actual Label")
	for ind, ax in enumerate(axs):
			ax.imshow(image_inputs[ind], cmap="Greys")
			pl = first_label if predicted_labels[ind] == 0.0 else second_label
			al = first_label if np.argmax(image_labels[ind], axis=0) == 0 else second_label
			ax.set(title="PL: {}\nAL: {}".format(pl, al))
			plt.setp(ax.get_xticklabels(), visible=False)
			plt.setp(ax.get_yticklabels(), visible=False)
			ax.tick_params(axis='both', which='both', length=0)
	plt.show()


def main():
	'''
	Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and 
	test your model for a number of epochs. We recommend that you train for 10 epochs and at most 25 epochs.
	For CS2470 students, you must train within 10 epochs.
	You should receive a final accuracy on the testing examples for cat and dog of >=70%.
	:return: None
	'''
	# load train and test data
	first_class = 3
	second_class = 5
	train_inputs, train_labels = get_data('CIFAR_data_compressed/train', first_class, second_class)
	test_inputs, test_labels = get_data('CIFAR_data_compressed/test', first_class, second_class)

	print("\ntrain set size is: ", train_inputs.shape, train_labels.shape)
	print("test set size is: ",test_inputs.shape, test_labels.shape)

	model = Model()

	for epoch in range(0, model.epoch):
		print("\n       -------------     EPOCH {}     -------------       ".format(epoch))
		train(model, train_inputs, train_labels)
	print("\n   -------------     ALL EPOCHS END     -------------    \n")

	test_accuracy = test(model, test_inputs, test_labels)
	print("Accuracy on test set is: {}".format(test_accuracy))

	# visualize 10 images
	sample_inputs = test_inputs[0:10]
	sample_labels = test_labels[0:10]
	sample_logits = model.call(sample_inputs, sample_labels)
	visualize_results(sample_inputs, sample_logits, sample_labels, 'cat', 'dog')


if __name__ == '__main__':
	main()
