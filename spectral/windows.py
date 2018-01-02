import tensorflow as tf
import numpy as np


def tukey(window_length, alpha=0.5, dtype=tf.float32):
	window_length = tf.convert_to_tensor(window_length, dtype=tf.int32)
	window_length_float = tf.cast(window_length, dtype=tf.float32)

	width = tf.floor(alpha * (window_length_float - 1) / 2.0)

	ones_start = width + 1
	ones_end = window_length_float - width - 1
	ones_length = tf.cast(ones_end - ones_start, tf.int32)

	n1 = tf.range(ones_start, dtype=dtype)
	n3 = tf.range(ones_end, window_length_float, dtype=dtype)

	w1 = 0.5 * (1 + tf.cos(np.pi * (-1 + 2.0 * n1 / alpha / (window_length_float - 1))))
	w2 = tf.ones([ones_length], dtype=dtype)
	w3 = 0.5 * (1 + tf.cos(np.pi * (-2.0 / alpha + 1 + 2.0 * n3 / alpha / (window_length_float - 1))))

	return tf.concat([w1, w2, w3], axis=0)
