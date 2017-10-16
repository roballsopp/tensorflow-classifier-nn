import tensorflow as tf

weights_init_seed = 2

def time_series_layers(inputs, reuse, data_format='channels_last'):
	inputs = tf.layers.conv1d(
		inputs,
		filters=32,
		kernel_size=16,
		strides=1,
		kernel_initializer=tf.contrib.layers.xavier_initializer(seed=weights_init_seed),
		data_format=data_format,
		activation=tf.nn.relu,
		name='time_series_layer_1',
		reuse=reuse
	)

	inputs = tf.layers.conv1d(
		inputs,
		filters=32,
		kernel_size=32,
		strides=1,
		kernel_initializer=tf.contrib.layers.xavier_initializer(seed=weights_init_seed),
		data_format=data_format,
		activation=tf.nn.relu,
		name='time_series_layer_2',
		reuse=reuse
	)

	inputs = tf.layers.conv1d(
		inputs,
		filters=16,
		kernel_size=64,
		strides=1,
		kernel_initializer=tf.contrib.layers.xavier_initializer(seed=weights_init_seed),
		data_format=data_format,
		activation=tf.nn.relu,
		name='time_series_layer_3',
		reuse=reuse
	)

	inputs = tf.layers.conv1d(
		inputs,
		filters=8,
		kernel_size=64,
		strides=1,
		kernel_initializer=tf.contrib.layers.xavier_initializer(seed=weights_init_seed),
		data_format=data_format,
		activation=tf.nn.relu,
		name='time_series_layer_4',
		reuse=reuse
	)

	return inputs

def spectrogram_layers(inputs, reuse=None, data_format='channels_last'):
	inputs = tf.layers.conv2d(
		inputs,
		filters=32,
		kernel_size=(2, 16),
		strides=(1, 1),
		kernel_initializer=tf.contrib.layers.xavier_initializer(seed=weights_init_seed),
		data_format=data_format,
		activation=tf.nn.relu,
		name='spectrogram_layer_1',
		reuse=reuse
	)

	inputs = tf.layers.conv2d(
		inputs,
		filters=32,
		kernel_size=(2, 32),
		strides=(2, 1),
		kernel_initializer=tf.contrib.layers.xavier_initializer(seed=weights_init_seed),
		data_format=data_format,
		activation=tf.nn.relu,
		name='spectrogram_layer_2',
		reuse=reuse
	)

	inputs = tf.layers.conv2d(
		inputs,
		filters=16,
		kernel_size=(2, 64),
		strides=(2, 1),
		kernel_initializer=tf.contrib.layers.xavier_initializer(seed=weights_init_seed),
		data_format=data_format,
		activation=tf.nn.relu,
		name='spectrogram_layer_3',
		reuse=reuse
	)

	inputs = tf.layers.conv2d(
		inputs,
		filters=8,
		kernel_size=(4, 64),
		strides=(4, 1),
		kernel_initializer=tf.contrib.layers.xavier_initializer(seed=weights_init_seed),
		data_format=data_format,
		activation=tf.nn.relu,
		name='spectrogram_layer_4',
		reuse=reuse
	)

	return inputs

class Model:
	def __init__(self, time_series_inputs, spectrogram_inputs, reuse=None, data_format='channels_last'):
		time_series_out = time_series_layers(time_series_inputs, reuse, data_format)
		spectrogram_out = spectrogram_layers(spectrogram_inputs, reuse, data_format)

		time_series_out = tf.expand_dims(time_series_out, axis=1)

		merged_outs = tf.concat([time_series_out, spectrogram_out], axis=1)

		final_out = tf.layers.conv2d(
			merged_outs,
			filters=1,
			kernel_size=(3, 84),
			strides=(3, 84),
			kernel_initializer=tf.contrib.layers.xavier_initializer(seed=weights_init_seed),
			data_format=data_format,
			name='output_layer',
			reuse=reuse
		)

		self._raw_outputs = tf.squeeze(final_out, [2, 3])

	def forward_prop(self):
		return tf.nn.sigmoid(self._raw_outputs)

	def loss(self, correct_labels, positive_weight=1600):
		batch_size = tf.cast(tf.shape(correct_labels)[0], tf.float32)

		negative_mask = tf.abs(correct_labels - 1)
		weights = (correct_labels * positive_weight) + negative_mask

		return tf.losses.sigmoid_cross_entropy(correct_labels, logits=self._raw_outputs, reduction=tf.losses.Reduction.SUM) / batch_size
