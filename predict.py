import tensorflow as tf
import numpy as np
import nn

def slide_predict(layers, X, modelPath):
	output = None

	graph = tf.Graph()
	with graph.as_default():
		input_layer_size, hidden_layer_size, num_labels = layers

		X_placeholder = tf.placeholder(tf.float32, shape=(None, input_layer_size), name='X')
		Theta1 = tf.Variable(nn.randInitializeWeights(input_layer_size, hidden_layer_size), name='Theta1')
		bias1 = tf.Variable(nn.randInitializeWeights(hidden_layer_size, 1), name='bias1')
		Theta2 = tf.Variable(nn.randInitializeWeights(hidden_layer_size, num_labels), name='Theta2')
		bias2 = tf.Variable(nn.randInitializeWeights(num_labels, 1), name='bias2')
		hypothesis = nn.forward_prop(X_placeholder, Theta1, bias1, Theta2, bias2)

		sess = tf.Session(graph=graph)
		saver = tf.train.Saver()
		init = tf.global_variables_initializer()
		sess.run(init)

		saver.restore(sess, modelPath)

		window_size = layers[0]

		pad_amount = (window_size * 2) - (X.shape[0] % window_size)
		X = np.pad(X, (pad_amount, 0), 'constant')

		for w in range(window_size):
			start = w
			end = -window_size + w
			X_shifted = X[start:end]
			X_matrix = X_shifted.reshape((-1, window_size))

			prediction = sess.run(hypothesis, feed_dict={X_placeholder: X_matrix})

			output = prediction if (output is None) else np.hstack((output, prediction))

		sess.close()

	output.shape = (X.size, -1)

	return output


def predict_window(output, i, window_size, tf_x, theta1, bias1, theta2, bias2):
	shift_size = tf.size(tf_x) - window_size
	tf_x_shifted = tf.slice(tf_x, [i], [shift_size], name='Shifted')
	tf_x_matrix = tf.reshape(tf_x_shifted, [-1, window_size], name='Matrix')
	prediction = nn.forward_prop(tf_x_matrix, theta1, bias1, theta2, bias2)
	output_stacked = tf.cond(tf.equal(i, 0), lambda: prediction, lambda: tf.concat(1, [output, prediction]))
	return [output_stacked, tf.add(i, 25), window_size, tf_x, theta1, bias1, theta2, bias2]


def slider(layers, x, model_path):
	input_layer_size, hidden_layer_size, num_labels = layers
	window_size = input_layer_size

	graph = tf.Graph()
	with graph.as_default():

		tf_x = tf.placeholder(tf.float32, name='X')
		theta1 = tf.Variable(nn.randInitializeWeights(input_layer_size, hidden_layer_size), name='Theta1')
		bias1 = tf.Variable(nn.randInitializeWeights(hidden_layer_size, 1), name='bias1')
		theta2 = tf.Variable(nn.randInitializeWeights(hidden_layer_size, num_labels), name='Theta2')
		bias2 = tf.Variable(nn.randInitializeWeights(num_labels, 1), name='bias2')

		tf_x_length = tf.size(tf_x)
		pad_amount = (window_size * 2) - (tf_x_length % window_size)
		tf_x_padded = tf.pad(tf_x, [[0, pad_amount]], 'CONSTANT')

		i = tf.constant(0)
		output_acc = tf.Variable(tf.zeros([0, num_labels], dtype=tf.float32))

		loop_vars = [
			output_acc,
			i, window_size,
			tf_x_padded,
			theta1,
			bias1,
			theta2,
			bias2
		]

		loop_var_shapes = [
			tf.TensorShape([None, None]),
			i.get_shape(),
			tf.TensorShape([]),
			tf_x.get_shape(),
			theta1.get_shape(),
			bias1.get_shape(),
			theta2.get_shape(),
			bias2.get_shape()
		]

		results = tf.while_loop((lambda o,i,w,x,t1,b1,t2,b2: tf.less(i, window_size)), predict_window, loop_vars, shape_invariants=loop_var_shapes)

		output = tf.reshape(results[0], [-1, num_labels])

		sess = tf.Session(graph=graph)
		saver = tf.train.Saver([theta1, bias1, theta2, bias2])
		saver.restore(sess, model_path)

		predictions = sess.run(output, feed_dict={tf_x: x})
		sess.close()

		return predictions
