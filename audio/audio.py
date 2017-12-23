import tensorflow as tf

# assumes a batch of 1d signals
def interpolate(x, new_width, channels_last=True):
	height_axis = 1 if channels_last else 2
	chan_axis = -1 if channels_last else 1
	num_chan = x.shape[chan_axis].value
	# create a "height" dim, because resize_images expects to be working in a 2d space
	x = tf.expand_dims(x, axis=height_axis)

	if not channels_last:
		x = tf.transpose(x, perm=[0, 2, 3, 1])

	interpolated_x = tf.image.resize_images(
		x,
		size=[num_chan, new_width],
		# TODO: bicubic vs bilinear
		method=tf.image.ResizeMethod.BILINEAR
	)

	if not channels_last:
		interpolated_x = tf.transpose(interpolated_x, perm=[0, 3, 1, 2])

	interpolated_x = tf.squeeze(interpolated_x, axis=[height_axis])

	return interpolated_x