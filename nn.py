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
		layer1_out = tf.layers.conv1d(
			inputs,
			filters=20,
			kernel_size=100,  # if input sample rate is 11025, 25 samples is ~2ms
			strides=1,
			padding='same',
			activation=tf.nn.relu,
			name='hidden_layer_1',
			reuse=reuse
		)

		pool1_out = tf.layers.max_pooling1d(layer1_out, 50, 1, padding='same', name='pooling_layer_1')  # pools the highest filter values over the next 50 samples

		layer2_out = tf.layers.conv1d(
			pool1_out,
			filters=5,
			kernel_size=50,
			strides=1,
			padding='same',
			activation=tf.nn.relu,
			name='hidden_layer_2',
			reuse=reuse
		)

		pool2_out = tf.layers.max_pooling1d(layer2_out, 50, 1, padding='same', name='pooling_layer_2')

		final_out = tf.layers.conv1d(
			pool2_out,
			filters=1,
			kernel_size=6,  # use 6 pooled frames to determine if this is a hit or not
			strides=1,
			padding='same',
			name='output_layer',
			reuse=reuse
		)

		self._raw_outputs = tf.squeeze(final_out)

	def forward_prop(self):
		return tf.nn.sigmoid(self._raw_outputs)

	def loss(self, correct_labels):
		batch_size = tf.cast(tf.shape(correct_labels)[0], tf.float32)

		positive_ratio = 1 / tf.reduce_mean(correct_labels)
		negative_mask = tf.abs(correct_labels - 1)
		weights = (correct_labels * positive_ratio) + negative_mask

		return tf.losses.sigmoid_cross_entropy(correct_labels, logits=self._raw_outputs, weights=weights, reduction=tf.losses.Reduction.SUM) / batch_size
