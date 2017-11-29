import tensorflow as tf

def transient_filter_initializer(shape, dtype=None, partition_info=None):
	batch, height, width, chan = shape
	half_height = round(height / 2)

	find_low = list(range(-1, -half_height - 1, -1))
	find_high = list(range(half_height, 0, -1))

	var = tf.convert_to_tensor([[find_low + find_high]] * width, dtype=dtype)
	var = tf.transpose(tf.reshape(var, [batch, width, height, chan]))

	var = tf.Print(var, [var], summarize=1000)

	return var

def create_layer(inputs, size):
	pad_amt = round(size / 2)

	inputs = tf.pad(inputs, [[0, 0], [0, 0], [pad_amt, pad_amt], [0, 0]], "CONSTANT")

	inputs = tf.layers.conv2d(
		inputs,
		filters=1,
		kernel_size=[1, size],
		strides=[1, 1],
		activation=None,
		use_bias=False,
		kernel_initializer=transient_filter_initializer,
		trainable=False,
		name='layer_' + str(size),
	)

	# inputs = inputs[:, :, :-pad_amt, :]
	#
	# print(inputs.get_shape().as_list())

	return inputs


def rms_normalize_per_band(spectrogram):
	band_rms = tf.sqrt(tf.reduce_mean(tf.square(spectrogram), axis=[2]))
	den = tf.expand_dims(band_rms, axis=1)

	return spectrogram / den


class Model:
	def __init__(self, inputs):

		stfts = tf.contrib.signal.stft(tf.squeeze(inputs, axis=[2]), frame_length=64, frame_step=1, fft_length=64, pad_end=True)
		inputs = tf.abs(stfts)

		print(inputs.get_shape().as_list())

		inputs = rms_normalize_per_band(inputs)

		outputs = create_layer(inputs, 500)

		print(outputs.get_shape().as_list())

		self._raw_outputs = outputs

	def get_savable_vars(self):
		return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dual_model')

	def get_raw(self):
		return self._raw_outputs

	def forward_prop(self):
		return tf.nn.sigmoid(self._raw_outputs)

	def loss(self, correct_labels):
		batch_size = tf.cast(tf.shape(correct_labels)[0], tf.float32)
		return tf.losses.sigmoid_cross_entropy(correct_labels, logits=self._raw_outputs, reduction=tf.losses.Reduction.SUM) / batch_size
