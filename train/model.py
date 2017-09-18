import tensorflow as tf

class Model:
	def __init__(self, inputs, reuse=None):
		layer1_out = tf.layers.conv1d(
			inputs,
			filters=64,
			kernel_size=32,  # if input sample rate is 11025, 25 samples is ~2ms
			strides=1,
			padding='same',
			activation=tf.nn.relu,
			name='hidden_layer_1',
			reuse=reuse
		)

		# pool1_out = tf.layers.max_pooling1d(layer1_out, 25, 1, padding='same', name='pooling_layer_1')  # pools the highest filter values over the next 50 samples

		layer2_out = tf.layers.conv1d(
			layer1_out,
			filters=20,
			kernel_size=128,
			strides=1,
			padding='same',
			activation=tf.nn.relu,
			name='hidden_layer_2',
			reuse=reuse
		)

		pool2_out = tf.layers.max_pooling1d(layer2_out, 25, 1, padding='same', name='pooling_layer_2')

		layer3_out = tf.layers.conv1d(
			pool2_out,
			filters=5,
			kernel_size=256,
			strides=1,
			padding='same',
			activation=tf.nn.relu,
			name='hidden_layer_3',
			reuse=reuse
		)

		pool3_out = tf.layers.max_pooling1d(layer3_out, 25, 1, padding='same', name='pooling_layer_3')

		final_out = tf.layers.conv1d(
			pool3_out,
			filters=1,
			kernel_size=150,  # use 6 pooled frames to determine if this is a hit or not
			strides=1,
			padding='same',
			name='output_layer',
			reuse=reuse
		)

		self._raw_outputs = tf.squeeze(final_out)

	def forward_prop(self):
		return tf.nn.sigmoid(self._raw_outputs)

	def loss(self, correct_labels, positive_weight=1600):
		batch_size = tf.cast(tf.shape(correct_labels)[0], tf.float32)

		negative_mask = tf.abs(correct_labels - 1)
		weights = (correct_labels * positive_weight) + negative_mask

		return tf.losses.sigmoid_cross_entropy(correct_labels, logits=self._raw_outputs, weights=weights, reduction=tf.losses.Reduction.SUM) / batch_size
