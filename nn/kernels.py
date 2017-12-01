import tensorflow as tf

def get_exp_kernel(lobe_size, sharpness=2.0, dtype=tf.float32):
	positive_lobe = tf.range(lobe_size, 0, -1, dtype=dtype)
	positive_lobe = tf.pow(positive_lobe, sharpness)
	negative_lobe = tf.reverse(positive_lobe, axis=[0]) * -1

	return tf.concat([negative_lobe, positive_lobe], axis=0)


def fill_shape_with_kernel(shape, kernel):
	kernel_height, kernel_width, channels_in, channnels_out = shape
	# just copy the kernel for each requested band, channel, and kernel
	kernel = tf.tile(kernel, multiples=[kernel_width * channels_in * channnels_out])
	# reshape backwards so we end up with the values in the right dims, then transpose to fit correct kernel dims
	kernel = tf.reshape(kernel, [channnels_out, channels_in, kernel_width, kernel_height])
	kernel = tf.transpose(kernel)

	return kernel

def transient_kernel(shape, dtype=tf.float32):
	lobe_size = round(shape[0] / 2)
	kernel = get_exp_kernel(lobe_size, sharpness=1.0, dtype=dtype)
	kernel = fill_shape_with_kernel(shape, kernel)

	return kernel

def peak_kernel(shape, dtype=tf.float32):
	lobe_size = round(shape[0] / 2)
	kernel = tf.convert_to_tensor(([1] * lobe_size) + ([-1] * lobe_size), dtype=dtype)
	kernel = fill_shape_with_kernel(shape, kernel)

	return kernel


def blur_kernel(shape, dtype=tf.float32):
	lobe_size = round(shape[0] / 2)
	left_lobe = tf.range(1, lobe_size + 1, dtype=dtype)
	right_lobe = tf.reverse(left_lobe, axis=[0])
	kernel = tf.concat([left_lobe, right_lobe], axis=0)
	kernel = fill_shape_with_kernel(shape, kernel)

	return kernel
