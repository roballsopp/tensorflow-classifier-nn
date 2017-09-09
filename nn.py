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
	def __init__(self, inputs, hidden_layers, output_size, reuse=None):
		for i, num_outs in enumerate(hidden_layers):
			inputs = tf.layers.dense(inputs=inputs, units=num_outs, activation=tf.nn.sigmoid, reuse=reuse, name='hidden_layer_' + str(i))

		self._raw_outputs = tf.layers.dense(inputs=inputs, units=output_size, reuse=reuse, name='output_layer')

	def forward_prop(self):
		return tf.nn.sigmoid(self._raw_outputs)

	def loss(self, correct_labels):
		return tf.losses.sigmoid_cross_entropy(correct_labels, logits=self._raw_outputs, reduction=tf.losses.Reduction.SUM)
