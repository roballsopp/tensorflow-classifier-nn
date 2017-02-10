import tensorflow as tf

def randInitializeWeights(numInputs, numOutputs):
	epsilon = 0.12
	return tf.random_uniform([numOutputs, numInputs], minval=-epsilon, maxval=epsilon, dtype=tf.float32)

def constructNN(layers):
	weights = []
	biases = []

	for l in range(len(layers) - 1):
		num_inputs = layers[l]
		num_outputs = layers[l+1]

		theta_name = 'Theta' + str(l)
		bias_name = 'bias' + str(l)

		theta = tf.Variable(randInitializeWeights(num_inputs, num_outputs), name=theta_name)
		bias = tf.Variable(randInitializeWeights(num_outputs, 1), name=bias_name)

		weights.append(theta)
		biases.append(bias)

	return tuple(weights), tuple(biases)

def forward_prop(a, weights, biases):
	for l in range(len(weights)):
		theta = weights[l]
		bias = biases[l]
		z = tf.matmul(a, tf.transpose(theta)) + bias
		a = tf.sigmoid(z)
	return a

def cost(X, y, weights, biases):
	y_num = tf.cast(y, tf.float32)
	m = tf.to_float(tf.shape(X)[0])
	a3 = forward_prop(X, weights, biases)
	y1 = -y_num * tf.log(a3)
	y0 = (1 - y_num) * tf.log(1 - a3)
	return tf.reduce_sum((y1 - y0) / m)

def regTerm(Theta1, Theta2, m, lam):
	regTheta1 = tf.reduce_sum(Theta1 ** 2)
	regTheta2 = tf.reduce_sum(Theta2 ** 2)
	regTerm = lam / (2*m) * (regTheta1 + regTheta2)

def evaluate(X_val, y_val, weights, biases):
	y_hyp = forward_prop(X_val, weights, biases)

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

	tf.summary.scalar('num_true_pos', num_true_pos)
	tf.summary.scalar('num_false_pos', num_false_pos)
	tf.summary.scalar('num_true_neg', num_true_neg)
	tf.summary.scalar('num_false_neg', num_false_neg)

	num_total_pos = tf.reduce_sum(tf.cast(predicted_labels, tf.float32))
	
	tf.summary.scalar('num_total_pos', num_total_pos)

	get_precision = lambda: (num_true_pos / num_total_pos)
	get_recall = lambda: (num_true_pos / (num_true_pos + num_false_neg))
	get_zero = lambda: tf.constant(0, dtype=tf.float32)

	precision = tf.cond(tf.cast(num_total_pos, tf.bool), get_precision, get_zero)
	recall = tf.cond(tf.cast((num_true_pos + num_false_neg), tf.bool), get_recall, get_zero)

	get_f1_number = lambda: (2 * precision * recall / (precision + recall))

	f1 = tf.cond(tf.cast((precision + recall), tf.bool), get_f1_number, get_zero)

	return [accuracy, precision, recall, f1]

# J = (sum(sum(y1 - y0)) / m) + regTerm