import tensorflow as tf
import nn.kernels
import nn.math


def find_peaks_1d(inputs, channels_last=True):
	if channels_last:
		# all upward sloping locations are positive, plateaus will be 0, and all down slopes are negative
		diff = inputs[:, 1:, :] - inputs[:, :-1, :]
		# all up slopes are 1, everywhere else is 0
		peaks = tf.cast(diff > 0, dtype=tf.float32)
		# put a -1 at all locations where a transition from a 1 to a 0 occurred
		# this effectively marks all locations where an onset stopped, and an offset or plateau began
		# a +1 will be at the locations where a plateau or down slope began to go up
		peaks = peaks[:, 1:, :] - peaks[:, :-1, :]
		# convert those -1s to 1s, and ignore the rest
		peaks = tf.maximum(peaks * -1, 0)
		# add back the two elements that were lost to the two diffing stages above
		peaks = tf.pad(peaks, [[0, 0], [1, 1],  [0, 0]])
		return peaks

	# all the same as above, just channels first
	diff = inputs[:, :, 1:] - inputs[:, :, :-1]
	peaks = tf.cast(diff > 0, dtype=tf.float32)
	peaks = peaks[:, :, 1:] - peaks[:, :, :-1]
	peaks = tf.maximum(peaks * -1, 0)
	peaks = tf.pad(peaks, [[0, 0], [0, 0], [1, 1]])

	return peaks


# TODO: handle channels first, and padding seems to be off
def smooth(inputs, size=128):
	inputs = tf.layers.average_pooling2d(
		inputs,
		pool_size=[size, 1],
		strides=[1, 1],
		padding='same'
	)

	return inputs


# TODO: handle channels first
def smooth_1d(inputs, size=128):
	inputs = tf.layers.average_pooling1d(
		inputs,
		pool_size=size,
		strides=1,
		padding='same'
	)

	return inputs


def maximize(inputs, size=128, channels_last=True):
	if channels_last:
		inputs = tf.pad(inputs, [[0, 0], [size, size - 1], [0, 0], [0, 0]], 'CONSTANT')
	else:
		inputs = tf.pad(inputs, [[0, 0], [0, 0], [size, size - 1], [0, 0]], 'CONSTANT')

	inputs = tf.layers.max_pooling2d(
		inputs,
		pool_size=[size, 1],
		strides=[1, 1],
		padding='valid'
	)

	if channels_last:
		return inputs[:, :-size, :, :]
	else:
		return inputs[:, :, :-size, :]
