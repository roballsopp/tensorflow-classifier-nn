import tensorflow as tf
import nn

def train(layers, data, folder = 'run1'):
	input_layer_size, hidden_layer_size, num_labels = layers
	X, y, X_val, y_val = data

	graph = tf.Graph()
	with graph.as_default():
		X_placeholder = tf.placeholder(tf.float32, shape=(None, input_layer_size), name='X')
		y_placeholder = tf.placeholder(tf.uint8, shape=(None, num_labels), name='y')
		Theta1 = tf.Variable(nn.randInitializeWeights(input_layer_size, hidden_layer_size), name='Theta1')
		bias1 = tf.Variable(nn.randInitializeWeights(hidden_layer_size, 1), name='bias1')
		Theta2 = tf.Variable(nn.randInitializeWeights(hidden_layer_size, num_labels), name='Theta2')
		bias2 = tf.Variable(nn.randInitializeWeights(num_labels, 1), name='bias2')
		cost = nn.cost(X_placeholder, y_placeholder, Theta1, bias1, Theta2, bias2)
		optimize = tf.train.GradientDescentOptimizer(0.6).minimize(cost)

		accuracy, precision, recall, f1 = nn.evaluate(X_placeholder, y_placeholder, Theta1, bias1, Theta2, bias2)

		cost_summary = tf.summary.scalar('cost', cost)
		accuracy_summary = tf.summary.scalar('accuracy', accuracy)
		precision_summary = tf.summary.scalar('precision', precision)
		recall_summary = tf.summary.scalar('recall', recall)
		f1_summary = tf.summary.scalar('f1', f1)
		summaries = tf.summary.merge_all()

		sess = tf.Session(graph=graph)
		saver = tf.train.Saver()
		init = tf.global_variables_initializer()
		sess.run(init)

		writer = tf.summary.FileWriter('./tmp/logs/' + folder, sess.graph)

		NUM_STEPS = 4000

		for step in range(NUM_STEPS):
			sess.run(optimize, feed_dict={X_placeholder: X, y_placeholder: y})
			if (step > 0) and ((step + 1) % 10 == 0):
				acc, prec, rec, f, summary = sess.run([accuracy, precision, recall, f1, summaries], feed_dict={X_placeholder: X_val, y_placeholder: y_val})
				writer.add_summary(summary, step)
				print('Step', step + 1, 'of', NUM_STEPS)

		save_path = saver.save(sess, './tmp/model_' + folder + '.ckpt')
		print("Model saved in file: %s" % save_path)
		sess.close()
