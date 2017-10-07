import tensorflow as tf

class Model:
	def __init__(self, inputs, reuse=None, data_format='channels_last'):
		inputs = tf.layers.conv2d(
			inputs,
			filters=16,
			kernel_size=(16, 3),  # if input sample rate is 11025, 25 samples is ~2ms
			strides=(1, 3),
			padding='same',
			data_format=data_format,
			activation=tf.nn.relu,
			name='hidden_layer_1',
			reuse=reuse
		)

		inputs = tf.layers.conv2d(
			inputs,
			filters=16,
			kernel_size=(16, 3),
			strides=(1, 2),
			padding='same',
			data_format=data_format,
			activation=tf.nn.relu,
			name='hidden_layer_2',
			reuse=reuse
		)

		inputs = tf.layers.conv2d(
			inputs,
			filters=5,
			kernel_size=(64, 6),
			strides=(1, 6),
			padding='same',
			data_format=data_format,
			activation=tf.nn.relu,
			name='hidden_layer_3',
			reuse=reuse
		)

		final_out = tf.layers.conv2d(
			inputs,
			filters=1,
			kernel_size=(100, 1),
			padding='same',
			data_format=data_format,
			name='output_layer',
			reuse=reuse
		)

		self._raw_outputs = tf.squeeze(final_out)

	def forward_prop(self):
		return tf.nn.sigmoid(self._raw_outputs)

	def loss(self, correct_labels, positive_weight=1000):
		batch_size = tf.cast(tf.shape(correct_labels)[0], tf.float32)

		negative_mask = tf.abs(correct_labels - 1)
		weights = (correct_labels * positive_weight) + negative_mask

		return tf.losses.sigmoid_cross_entropy(correct_labels, logits=self._raw_outputs, weights=weights, reduction=tf.losses.Reduction.SUM) / batch_size
