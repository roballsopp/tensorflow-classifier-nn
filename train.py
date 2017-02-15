import tensorflow as tf
import numpy as np
import nn

def train(layers, data, folder='run1'):
	np.random.shuffle(data)

	graph = tf.Graph()
	with graph.as_default():
		X_train = tf.constant(data['X'][:-1000])
		y_train = tf.constant(data['y'][:-1000])
		X_val = tf.constant(data['X'][-1000:])
		y_val = tf.constant(data['y'][-1000:])

		net = nn.FullyConnected(layers)

		hyp_train = net.forward_prop(X_train)
		hyp_val = net.forward_prop(X_val)

		train_cost = nn.cross_entropy(hyp_train, y_train)
		val_cost = nn.cross_entropy(hyp_val, y_val)

		optimize = tf.train.AdamOptimizer().minimize(train_cost)

		metrics_train = nn.evaluate(hyp_train, y_train)
		metrics_val = nn.evaluate(hyp_val, y_val)

		tf.summary.scalar('cost_train', train_cost)
		tf.summary.scalar('cost_val', val_cost)

		tf.summary.scalar('accuracy_train', metrics_train['accuracy'])

		for metric_name in metrics_val:
			tf.summary.scalar(metric_name, metrics_val[metric_name])

		summaries = tf.summary.merge_all()

		sess = tf.Session(graph=graph)

		session_saver = net.get_saver()
		summary_writer = tf.summary.FileWriter('./tmp/logs/' + folder)

		init = tf.global_variables_initializer()
		sess.run(init)

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
