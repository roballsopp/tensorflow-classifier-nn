import tensorflow as tf
import numpy as np
import nn

def train(layers, data, folder='run1'):
	np.random.shuffle(data)

	graph = tf.Graph()
	with graph.as_default():
		X = tf.constant(data['X'][:-1000])
		y = tf.constant(data['y'][:-1000])
		X_val = tf.constant(data['X'][-1000:])
		y_val = tf.constant(data['y'][-1000:])

		weights, biases = nn.constructNN(layers)

		train_cost = nn.cost(X, y, weights, biases)

		optimize = tf.train.AdamOptimizer().minimize(train_cost)

		metrics = nn.evaluate(X_val, y_val, weights, biases)

		tf.summary.scalar('cost', train_cost)
		tf.summary.scalar('accuracy', metrics['accuracy'])
		tf.summary.scalar('precision', metrics['precision'])
		tf.summary.scalar('recall', metrics['recall'])
		tf.summary.scalar('f1', metrics['f1'])
		tf.summary.scalar('num_true_pos', metrics['num_true_pos'])
		tf.summary.scalar('num_false_pos', metrics['num_false_pos'])
		tf.summary.scalar('num_true_neg', metrics['num_true_neg'])
		tf.summary.scalar('num_false_neg', metrics['num_false_neg'])

		summaries = tf.summary.merge_all()

		sess = tf.Session(graph=graph)
		session_saver = tf.train.Saver(list(weights) + list(biases))
		init = tf.global_variables_initializer()
		sess.run(init)

		summary_writer = tf.summary.FileWriter('./tmp/logs/' + folder)

		NUM_STEPS = 4000

		for step in range(NUM_STEPS):
			sess.run(optimize)
			if (step > 0) and ((step + 1) % 10 == 0):
				summary_results = sess.run(summaries)
				summary_writer.add_summary(summary_results, step)
				print('Step', step + 1, 'of', NUM_STEPS)

		save_path = session_saver.save(sess, './tmp/model_' + folder + '.ckpt')
		print("Model saved in file: %s" % save_path)
		sess.close()
