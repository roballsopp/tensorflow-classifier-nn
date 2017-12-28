import tensorflow as tf


def rms(inputs, axis=None, non_zero=False):
	inputs_squared = tf.square(inputs)
	if non_zero:
		squared_mean = tf.reduce_sum(inputs_squared, keep_dims=True, axis=axis) / tf.count_nonzero(inputs_squared, dtype=tf.float32)
	else:
		squared_mean = tf.reduce_mean(inputs_squared, keep_dims=True, axis=axis)

	rms = tf.sqrt(squared_mean)

	return rms


def rms_normalize(inputs, axis=None, non_zero=False):
	return inputs / rms(inputs, axis, non_zero)


def mean_normalize(inputs, axis=None, non_zero=False):
	if non_zero:
		mean = tf.reduce_sum(inputs, keep_dims=True, axis=axis) / tf.count_nonzero(inputs, dtype=tf.float32)
	else:
		mean = tf.reduce_mean(inputs, keep_dims=True, axis=axis)

	return inputs / mean


def abs_mean_normalize(inputs, axis=None, non_zero=False):
	inputs_abs = tf.abs(inputs)
	if non_zero:
		mean = tf.reduce_sum(inputs_abs, keep_dims=True, axis=axis) / tf.count_nonzero(inputs_abs, dtype=tf.float32)
	else:
		mean = tf.reduce_mean(inputs_abs, keep_dims=True, axis=axis)

	return inputs / mean


def normalize(inputs, axis=None):
	maximum = tf.reduce_max(tf.abs(inputs), keep_dims=True, axis=axis)
	return inputs / maximum


def fix_divide_by_zero(inputs):
	return tf.where(tf.is_nan(inputs), x=tf.zeros(inputs.shape, dtype=inputs.dtype), y=inputs)
