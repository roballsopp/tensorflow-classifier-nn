import tensorflow as tf

def swish(x):
	return x * tf.nn.sigmoid(x)

def get_l2_regularization(lam=0.0):
	weights_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dual_model/weights')

	total_l2 = tf.constant(0, dtype=tf.float32)

	for weights in weights_list:
		total_l2 += tf.nn.l2_loss(weights)

	return lam * total_l2

def conv1d_bn(inputs, name, training=True, reuse=False, data_format='channels_last', **kwargs):
	with tf.variable_scope('weights'):
		conv_out = tf.layers.conv1d(inputs, **kwargs, data_format=data_format, reuse=reuse, name=name)

	example_maxes = tf.reduce_max(tf.abs(conv_out), axis=[1, 2])
	return conv_out / tf.reshape(example_maxes, [-1, 1, 1])

def conv2d_bn(inputs, name, training=True, reuse=False, data_format='channels_last', **kwargs):
	with tf.variable_scope('weights'):
		conv_out = tf.layers.conv2d(inputs, **kwargs, data_format=data_format, reuse=reuse, name=name)

	example_maxes = tf.reduce_max(tf.abs(conv_out), axis=[1, 2, 3])
	return conv_out / tf.reshape(example_maxes, [-1, 1, 1, 1])

weights_init_seed = 5
activation = swish

def time_series_layers(inputs, training, reuse, data_format='channels_last'):
	inputs = conv2d_bn(
		inputs,
		filters=512,
		kernel_size=(1, 32),
		strides=(1, 1),
		kernel_initializer=tf.contrib.layers.xavier_initializer(seed=weights_init_seed),
		data_format=data_format,
		activation=activation,
		name='time_series_layer_1',
		reuse=reuse,
		training=training
	)

	inputs = conv2d_bn(
		inputs,
		filters=64,
		kernel_size=(1, 32),
		strides=(1, 1),
		kernel_initializer=tf.contrib.layers.xavier_initializer(seed=weights_init_seed),
		data_format=data_format,
		activation=activation,
		name='time_series_layer_2',
		reuse=reuse,
		training=training
	)

	inputs = conv2d_bn(
		inputs,
		filters=512,
		kernel_size=(1, 450),
		strides=(1, 1),
		kernel_initializer=tf.contrib.layers.xavier_initializer(seed=weights_init_seed),
		data_format=data_format,
		activation=activation,
		name='time_series_layer_3',
		reuse=reuse,
		training=training
	)

	return inputs


def spectrogram_layers(inputs, training, reuse, data_format='channels_last'):
	inputs = conv2d_bn(
		inputs,
		filters=32,
		kernel_size=(2, 8),
		strides=(1, 1),
		kernel_initializer=tf.contrib.layers.xavier_initializer(seed=weights_init_seed),
		data_format=data_format,
		activation=activation,
		name='spectrogram_layer_1',
		reuse=reuse,
		training=training
	)

	inputs = conv2d_bn(
		inputs,
		filters=32,
		kernel_size=(1, 32),
		strides=(1, 1),
		kernel_initializer=tf.contrib.layers.xavier_initializer(seed=weights_init_seed),
		data_format=data_format,
		activation=activation,
		name='spectrogram_layer_2',
		reuse=reuse,
		training=training
	)

	inputs = conv2d_bn(
		inputs,
		filters=64,
		kernel_size=(32, 218),
		strides=(1, 1),
		kernel_initializer=tf.contrib.layers.xavier_initializer(seed=weights_init_seed),
		data_format=data_format,
		activation=activation,
		name='spectrogram_layer_3',
		reuse=reuse,
		training=training
	)

	return inputs

def get_fft(time_series_inputs, channels_last=False):
	channel_axis = -1 if channels_last else 1

	stfts = tf.contrib.signal.stft(
		tf.squeeze(time_series_inputs, axis=[1, 2]),
		frame_length=64, frame_step=1, fft_length=64, pad_end=True
	)

	return tf.expand_dims(tf.abs(stfts), axis=channel_axis)


class Model:
	def __init__(self, time_series_inputs, training, reuse=False, channels_last=False):
		data_format = 'channels_last' if channels_last else 'channels_first'
		height_axis = 1 if channels_last else 2

		with tf.variable_scope("dual_model"):
			time_series_out = time_series_layers(time_series_inputs, training, reuse, data_format)
			spectrogram_inputs = get_fft(time_series_inputs, channels_last=channels_last)
			spectrogram_out = spectrogram_layers(spectrogram_inputs, training, reuse, data_format)

			time_series_out = tf.expand_dims(time_series_out, axis=height_axis)

			merged_outs = tf.concat([time_series_out, spectrogram_out], axis=height_axis, name='final_input_merge')

			final_out = tf.layers.conv2d(
				merged_outs,
				filters=1,
				kernel_size=(2, 1),
				strides=(1, 1),
				kernel_initializer=tf.contrib.layers.xavier_initializer(seed=weights_init_seed),
				data_format=data_format,
				name='output_layer',
				reuse=reuse
			)

		self._raw_outputs = tf.squeeze(final_out, [2, 3])

	def get_savable_vars(self):
		return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dual_model')

	def forward_prop(self):
		return tf.nn.sigmoid(self._raw_outputs)

	def loss(self, correct_labels, lam=0.001):
		batch_size = tf.cast(tf.shape(correct_labels)[0], tf.float32)

		# l2 = get_l2_regularization(lam)

		return tf.losses.sigmoid_cross_entropy(correct_labels, logits=self._raw_outputs, reduction=tf.losses.Reduction.SUM) / batch_size
