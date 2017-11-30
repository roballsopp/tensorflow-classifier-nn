import tensorflow as tf

def get_exp_filter(lobe_size, sharpness=2.0, dtype=tf.float32):
	positive_lobe = tf.range(lobe_size, 0, -1, dtype=dtype)
	positive_lobe = tf.pow(positive_lobe, sharpness)
	negative_lobe = tf.reverse(positive_lobe, axis=[0]) * -1

	return tf.concat([negative_lobe, positive_lobe], axis=0)


def transient_filter_initializer(shape, dtype=tf.float32, partition_info=None):
	time_len, band_width, chan_in, num_filters = shape
	lobe_size = round(time_len / 2)

	var = get_exp_filter(lobe_size, sharpness=1.0, dtype=dtype)
	# jut copy the filter for each requested band, channel, and filter for now
	var = tf.tile(var, multiples=[band_width * chan_in * num_filters])

	# reshape backwards so we end up with the values in the right dims, then transpose to fit correct filter dims
	var = tf.reshape(var, [num_filters, chan_in, band_width, time_len])
	var = tf.transpose(var)

	return var

def create_layer(inputs, size, channels_last=True):
	pad_amt = round(size / 2)

	if channels_last:
		inputs = tf.pad(inputs, [[0, 0], [pad_amt, pad_amt], [0, 0], [0, 0]], "CONSTANT")
	else:
		inputs = tf.pad(inputs, [[0, 0], [0, 0], [pad_amt, pad_amt], [0, 0]], "CONSTANT")

	data_format = 'channels_last' if channels_last else 'channels_first'

	inputs = tf.layers.conv2d(
		inputs,
		filters=1,
		kernel_size=[size, 33],
		strides=[1, 1],
		activation=tf.nn.relu,
		use_bias=False,
		data_format=data_format,
		kernel_initializer=transient_filter_initializer,
		trainable=False,
		name='layer_1',
	)

	print(inputs.get_shape().as_list())

	# inputs = inputs[:, :, :-pad_amt, :]
	#
	print(inputs.get_shape().as_list())

	return inputs


def rms_normalize_per_band(spectrogram, channels_last=True):
	band_axis = 1 if channels_last else 2
	band_rms = tf.sqrt(tf.reduce_mean(tf.square(spectrogram), axis=[band_axis]))
	den = tf.expand_dims(band_rms, axis=1)

	return spectrogram / den


class Model:
	def __init__(self, inputs, channels_last=True):
		fft_size = 64
		num_output_bands = int((fft_size / 2) + 1)

		stfts = tf.contrib.signal.stft(inputs, frame_length=fft_size, frame_step=1, fft_length=fft_size, pad_end=True)
		inputs = tf.abs(stfts)

		if channels_last:
			inputs = tf.reshape(inputs, [-1, num_output_bands, 1])

		# create the "batch" dim, even though there is only one example
		inputs = tf.expand_dims(inputs, axis=0)

		inputs = rms_normalize_per_band(inputs, channels_last=channels_last)
		outputs = create_layer(inputs, 500, channels_last=channels_last)

		# remove "batch" dim, and band dim
		inputs = tf.squeeze(outputs, axis=[0, 2])
		# reshape back into channels first, TODO: might need to be a transpose with multi-channel
		self._raw_outputs = tf.reshape(inputs, [1, -1])

	def get_savable_vars(self):
		return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dual_model')

	def get_raw(self):
		return self._raw_outputs

	def forward_prop(self):
		return tf.nn.sigmoid(self._raw_outputs)

	def loss(self, correct_labels):
		batch_size = tf.cast(tf.shape(correct_labels)[0], tf.float32)
		return tf.losses.sigmoid_cross_entropy(correct_labels, logits=self._raw_outputs, reduction=tf.losses.Reduction.SUM) / batch_size
