import tensorflow as tf
import nn.kernels
import nn.math

def get_data_format_string(channels_last):
	return 'NHWC' if channels_last else 'NCHW'

def find_peaks(inputs, channels_last=True):
	data_format = get_data_format_string(channels_last)

	peak_filter = tf.nn.convolution(
		inputs,
		filter=nn.kernels.peak_kernel([2, 1, 1, 1]),
		padding='SAME',
		data_format=data_format,
		name='peak_filter_1',
	)

	peaks = peak_filter / tf.abs(peak_filter)
	peaks = tf.minimum(nn.math.fix_divide_by_zero(peaks), 0)

	peaks = tf.nn.convolution(
		peaks,
		filter=nn.kernels.peak_kernel([2, 1, 1, 1]),
		padding='SAME',
		data_format=data_format,
		name='peak_filter_2',
	)

	peaks = tf.maximum(peaks * -1, 0)

	return peaks


def smooth(inputs, size=128):
	inputs = tf.layers.average_pooling2d(
		inputs,
		pool_size=[size, 1],
		strides=[1, 1],
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
