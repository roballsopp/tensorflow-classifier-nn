import tensorflow as tf
import numpy as np

ALMOST_ZERO = np.nextafter(np.float32(0), np.float32(1))
ALMOST_ONE = np.nextafter(np.float32(1), np.float32(0))

class Layer:
	def __init__(self, w, b):
		self.w = w
		self.b = b

def randInitializeWeights(numInputs, numOutputs):
	epsilon = 0.12
	return tf.random_uniform([numOutputs, numInputs], minval=-epsilon, maxval=epsilon, dtype=tf.float32)

def _construct_layers(layer_defs):
	layers = []

	for l in range(len(layer_defs) - 1):
		num_inputs = layer_defs[l]
		num_outputs = layer_defs[l + 1]

		theta_name = 'Theta' + str(l)
		bias_name = 'bias' + str(l)

		theta = tf.Variable(randInitializeWeights(num_inputs, num_outputs), name=theta_name)
		bias = tf.Variable(randInitializeWeights(num_outputs, 1), name=bias_name)

		layers.append(Layer(theta, bias))

	return tuple(layers)

class FullyConnected:
	def __init__(self, layer_defs):
		self.layers = _construct_layers(layer_defs)

	def forward_prop(self, a):
		for layer in self.layers:
			z = tf.matmul(a, tf.transpose(layer.w)) + layer.b
			a = tf.sigmoid(z)
		return a

	def cross_entropy(self, X, y):
		y_num = tf.cast(y, tf.float32)
		m = tf.to_float(tf.shape(X)[0])
		a3 = self.forward_prop(X)
		# clipping prevents underflow errors (log(0))
		y1 = y_num * -tf.log(tf.clip_by_value(a3, ALMOST_ZERO, 1))
		y0 = (1 - y_num) * tf.log(1 - tf.clip_by_value(a3, 0, ALMOST_ONE))

		return tf.reduce_sum((y1 - y0) / m)

	def evaluate(self, X_val, y_val):
		y_hyp = self.forward_prop(X_val)

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

	def get_saver(self):
		variables_to_save = []

		for layer in self.layers:
			variables_to_save.append(layer.w)
			variables_to_save.append(layer.b)

		return tf.train.Saver(variables_to_save)
