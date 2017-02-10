import tensorflow as tf
import numpy as np
import nn

def train(layers, data, folder = 'run1'):
	num_features = layers[0]
	num_labels = layers[-1]

	np.random.shuffle(data)

	X = data['X'][:-1000]
	y = data['y'][:-1000]
	X_val = data['X'][-1000:]
	y_val = data['y'][-1000:]

	graph = tf.Graph()
	with graph.as_default():
		X_placeholder = tf.placeholder(tf.float32, shape=(None, num_features), name='X')
		y_placeholder = tf.placeholder(tf.uint8, shape=(None, num_labels), name='y')
		weights, biases = nn.constructNN(layers)
		cost = nn.cost(X_placeholder, y_placeholder, weights, biases)
		optimize = tf.train.GradientDescentOptimizer(1.5).minimize(cost)

		accuracy, precision, recall, f1 = nn.evaluate(X_placeholder, y_placeholder, weights, biases)

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
