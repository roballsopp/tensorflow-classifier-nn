import tensorflow as tf


# take a 1d kernel, and expand its dimensions for use in a 1d convolution op
# this will output a tensor of rank 3
def expand_1d(kernel, channels_in=1, channels_out=1):
	rank_check = tf.assert_rank(kernel, 1, message='Kernel must have rank 1 for use with expand_kernel_1d')

	kernel_height = kernel.shape[0].value

	with tf.control_dependencies([rank_check]):
		# just copy the kernel for each channel in and out
		kernel = tf.tile(kernel, multiples=[channels_in * channels_out])
		# reshape backwards so we end up with the values in the right dims, then transpose to fit correct kernel dims
		kernel = tf.reshape(kernel, [channels_out, channels_in, kernel_height])
		kernel = tf.transpose(kernel)

	return kernel


# take a 1d or 2d kernel, and expand its dimensions for use in a 2d convolution op
# this will output a tensor of rank 4
def expand_2d(kernel, width=None, channels_in=1, channels_out=1):
	rank_check = tf.assert_rank_in(kernel, [1, 2], message='Kernel must have rank 1 or 2 for use with expand_kernel_2d')

	kernel_rank = kernel.shape.ndims
	kernel_height = kernel.shape[0].value

	with tf.control_dependencies([rank_check]):
		if kernel_rank == 1:
			if width is None:
				raise ValueError('Width parameter required when kernel rank is 1')

			# just copy the kernel for each requested row and each channel in and out
			kernel = tf.tile(kernel, multiples=[width * channels_in * channels_out])
		else:
			# just copy the kernel for each channel in and out
			width = kernel.shape[0].value
			kernel = tf.tile(kernel, multiples=[channels_in * channels_out])

	# reshape backwards so we end up with the values in the right dims, then transpose to fit correct kernel dims
	kernel = tf.reshape(kernel, [channels_out, channels_in, width, kernel_height])
	kernel = tf.transpose(kernel)

	return kernel
