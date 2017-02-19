import tensorflow as tf
import numpy as np
import nn


def slider(layers, x, model_path, skip=50):
	input_layer_size, hidden_layer_size, num_labels = layers
	slice_size = ((x.size // input_layer_size) * input_layer_size) - input_layer_size

	graph = tf.Graph()
	with graph.as_default():

		tf_x = tf.placeholder(tf.float32, name='X')
		net = nn.FullyConnected(layers)

		output_acc = []

		for i in range(0, input_layer_size, skip):
			tf_x_shifted = tf.slice(tf_x, [i], [slice_size])
			tf_x_matrix = tf.reshape(tf_x_shifted, [-1, input_layer_size])
			prediction = net.forward_prop(tf_x_matrix)
			output_acc.append(prediction)

		output = tf.reshape(tf.concat(1, output_acc), [-1, num_labels])

		sess = tf.Session(graph=graph)
		saver = net.get_saver()
		saver.restore(sess, model_path)

		predictions = sess.run(output, feed_dict={tf_x: x})
		sess.close()

		return predictions

