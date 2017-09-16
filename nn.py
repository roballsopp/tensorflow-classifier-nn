import tensorflow as tf

def evaluate(y_hyp, y_val):
	correct_labels = tf.cast(y_val, tf.bool)
	predicted_labels = tf.cast(tf.round(y_hyp), tf.bool)

	correct_prediction = tf.equal(correct_labels, predicted_labels)

	true_pos = tf.logical_and(correct_labels, predicted_labels)
	false_pos = tf.logical_and(tf.logical_not(correct_labels), predicted_labels)
	true_neg = tf.logical_and(tf.logical_not(correct_labels), tf.logical_not(predicted_labels))
	false_neg = tf.logical_and(correct_labels, tf.logical_not(predicted_labels))

	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	num_true_pos = tf.reduce_sum(tf.cast(true_pos, tf.float32))
	num_false_pos = tf.reduce_sum(tf.cast(false_pos, tf.float32))
	num_true_neg = tf.reduce_sum(tf.cast(true_neg, tf.float32))
	num_false_neg = tf.reduce_sum(tf.cast(false_neg, tf.float32))

	num_total_pos = tf.reduce_sum(tf.cast(predicted_labels, tf.float32))

	get_precision = lambda: (num_true_pos / num_total_pos)
	get_recall = lambda: (num_true_pos / (num_true_pos + num_false_neg))
	get_zero = lambda: tf.constant(0, dtype=tf.float32)

	precision = tf.cond(tf.cast(num_total_pos, tf.bool), get_precision, get_zero)
	recall = tf.cond(tf.cast((num_true_pos + num_false_neg), tf.bool), get_recall, get_zero)

	get_f1_number = lambda: (2 * precision * recall / (precision + recall))

	f1 = tf.cond(tf.cast((precision + recall), tf.bool), get_f1_number, get_zero)

	return {
		'accuracy': accuracy,
		'precision': precision,
		'recall': recall,
		'f1': f1,
		'num_true_pos': num_true_pos,
		'num_false_pos': num_false_pos,
		'num_true_neg': num_true_neg,
		'num_false_neg': num_false_neg
	}

class Net:
	def __init__(self, inputs, reuse=None):
		num_filters = 10
		filter_size = 25  # if input sample rate is 11025, 25 samples is ~2ms

		layer1_out = tf.layers.conv1d(
			inputs,
			num_filters,
			filter_size,
			strides=1,
			padding='same',
			activation=tf.nn.relu,
			name='hidden_conv_layer',
			reuse=reuse
		)

		layer1_shape = tf.shape(layer1_out)

		layer2_input = tf.reshape(layer1_out, (-1, layer1_shape[1] * layer1_shape[2], 1))

		final_filter_size = num_filters * 150  # * 150 means 150 samples worth of audio will be considered to predict this sample
		strides = num_filters * 1  # * 10 means we will jump 10 samples for every prediction, reducing our output size by a factor of 10

		self._raw_outputs = tf.squeeze(tf.layers.conv1d(
			layer2_input,
			1,
			final_filter_size,
			strides=strides,
			padding='same',
			name='output_layer',
			reuse=reuse
		))

	def forward_prop(self):
		return tf.nn.sigmoid(self._raw_outputs)

	def loss(self, correct_labels):
		batch_size = tf.cast(tf.shape(correct_labels)[0], tf.float32)

		positive_ratio = 1 / tf.reduce_mean(correct_labels)
		negative_mask = tf.abs(correct_labels - 1)
		weights = (correct_labels * positive_ratio) + negative_mask

		return tf.losses.sigmoid_cross_entropy(correct_labels, logits=self._raw_outputs, weights=weights, reduction=tf.losses.Reduction.SUM) / batch_size
