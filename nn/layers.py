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