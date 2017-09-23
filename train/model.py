import tensorflow as tf

class Model:
	def __init__(self, inputs, reuse=None):
		layer1_out = tf.layers.conv1d(
			inputs,
			filters=64,
			kernel_size=16,  # if input sample rate is 11025, 25 samples is ~2ms
			padding='same',
			activation=tf.nn.relu,
			name='hidden_layer_1',
			reuse=reuse
		)

		layer2_out = tf.layers.conv1d(
			layer1_out,
			filters=20,
			kernel_size=64,
			padding='same',
			activation=tf.nn.relu,
			name='hidden_layer_2',
			reuse=reuse
		)

		layer3_out = tf.layers.conv1d(
			layer2_out,
			filters=5,
			kernel_size=64,
			padding='same',
			activation=tf.nn.relu,
			name='hidden_layer_3',
			reuse=reuse
		)

		final_out = tf.layers.conv1d(
			layer3_out,
			filters=1,
			kernel_size=100,
			padding='same',
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
