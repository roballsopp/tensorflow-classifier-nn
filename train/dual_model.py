import tensorflow as tf

def swish(x):
	return x * tf.nn.sigmoid(x)

def conv1d_bn(inputs, name, training=True, reuse=False, data_format='channels_last', **kwargs):
	conv_out = tf.layers.conv1d(inputs, **kwargs, data_format=data_format, reuse=reuse, name=name)

	norm_axis = -1 if data_format == 'channels_last' else 1

	return tf.layers.batch_normalization(conv_out, axis=norm_axis, training=training, reuse=reuse, fused=True, name=name + '_bn')

def conv2d_bn(inputs, name, training=True, reuse=False, data_format='channels_last', **kwargs):
	conv_out = tf.layers.conv2d(inputs, **kwargs, data_format=data_format, reuse=reuse, name=name)

	norm_axis = -1 if data_format == 'channels_last' else 1

	return tf.layers.batch_normalization(conv_out, axis=norm_axis, training=training, reuse=reuse, fused=True, name=name + '_bn')

weights_init_seed = 5
activation = swish

def time_series_layers(inputs, training, reuse, data_format='channels_last'):
	inputs = conv1d_bn(
		inputs,
		filters=32,
		kernel_size=8,
		strides=1,
		kernel_initializer=tf.contrib.layers.xavier_initializer(seed=weights_init_seed),
		data_format=data_format,
		activation=activation,
		name='time_series_layer_1',
		reuse=reuse,
		training=training
	)

	inputs = conv1d_bn(
		inputs,
		filters=32,
		kernel_size=32,
		strides=1,
		kernel_initializer=tf.contrib.layers.xavier_initializer(seed=weights_init_seed),
		data_format=data_format,
		activation=activation,
		name='time_series_layer_2',
		reuse=reuse,
		training=training
	)

	inputs = conv1d_bn(
		inputs,
		filters=64,
		kernel_size=218,
		strides=1,
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

class Model:
	def __init__(self, time_series_inputs, spectrogram_inputs, training, reuse=False, channels_last=False):
		data_format = 'channels_last' if channels_last else 'channels_first'
		height_axis = 1 if channels_last else 2

		with tf.variable_scope("dual_model"):
			time_series_out = time_series_layers(time_series_inputs, training, reuse, data_format)
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

	def loss(self, correct_labels, positive_weight=1600):
		batch_size = tf.cast(tf.shape(correct_labels)[0], tf.float32)

		negative_mask = tf.abs(correct_labels - 1)
		weights = (correct_labels * positive_weight) + negative_mask

		return tf.losses.sigmoid_cross_entropy(correct_labels, logits=self._raw_outputs, reduction=tf.losses.Reduction.SUM) / batch_size
