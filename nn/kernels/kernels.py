import tensorflow as tf


# generates a kernel like [-1, -2, -3, 3, 2, 1]. sharpness controls how sharp the peak is
# sharpness of 2.0 will make a kernel like [-1, -4, -9, 9, 4, 1]
# sharpness of 0.0 (default) makes it flat [-1, -1, -1, 1, 1, 1]
# TODO, this won't handle an odd sized kernel
def diff(height, sharpness=0.0, dtype=tf.float32):
	lobe_size = round(height / 2)

	positive_lobe = tf.range(lobe_size, 0, -1, dtype=dtype)
	positive_lobe = tf.pow(positive_lobe, sharpness)
	negative_lobe = tf.reverse(positive_lobe, axis=[0]) * -1

	kernel = tf.concat([negative_lobe, positive_lobe], axis=0)

	return kernel


# TODO, this won't handle an odd sized kernel
def blur(height, dtype=tf.float32):
	lobe_size = round(height / 2)
	left_lobe = tf.range(1, lobe_size + 1, dtype=dtype)
	right_lobe = tf.reverse(left_lobe, axis=[0])
	kernel = tf.concat([left_lobe, right_lobe], axis=0)

	return kernel
