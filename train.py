import tensorflow as tf
import numpy as np
import nn

def every_n_steps(n, step, callback):
	if (step > 0) and ((step + 1) % n == 0):
		callback(step)

def eval_train(metrics, cost):
	return tf.summary.merge([
		tf.summary.scalar('cost_train', cost),
		tf.summary.scalar('accuracy_train', metrics['accuracy'])
	])

def eval_val(metrics, cost):
	summaries = [tf.summary.scalar('cost_val', cost)]

	for metric_name in metrics:
		summary = tf.summary.scalar(metric_name, metrics[metric_name])
		summaries.append(summary)

	return tf.summary.merge(summaries)

def train(layers, data, folder='run1'):
	num_features = layers[0]
	num_labels = layers[-1]

	np.random.shuffle(data)

	X_train = data['X'][:-1000]
	y_train = data['y'][:-1000]
	X_val = data['X'][-1000:]
	y_val = data['y'][-1000:]

	graph = tf.Graph()
	with graph.as_default():
		X_pl = tf.placeholder(tf.float32, shape=(None, num_features))
		y_pl = tf.placeholder(tf.uint8, shape=(None, num_labels))

		net = nn.FullyConnected(layers)

		hyp = net.forward_prop(X_pl)
		cost = nn.cross_entropy(hyp, y_pl)
		optimize = tf.train.AdamOptimizer().minimize(cost)

		metrics = nn.evaluate(hyp, y_pl)

		train_summaries = eval_train(metrics, cost)
		val_summaries = eval_val(metrics, cost)

		init = tf.global_variables_initializer()

		with tf.Session() as sess:
			session_saver = net.get_saver()
			summary_writer = tf.summary.FileWriter('./tmp/logs/' + folder)
			sess.run(init)

			NUM_STEPS = 4000

			def add_summary(step):
				train_results = sess.run(train_summaries, feed_dict={X_pl: X_train, y_pl: y_train})
				val_results = sess.run(val_summaries, feed_dict={X_pl: X_val, y_pl: y_val})
				summary_writer.add_summary(train_results, step)
				summary_writer.add_summary(val_results, step)
				print('Step', step + 1, 'of', NUM_STEPS)

			for step in range(NUM_STEPS):
				sess.run(optimize, feed_dict={X_pl: X_train, y_pl: y_train})
				every_n_steps(10, step, add_summary)

			save_path = session_saver.save(sess, './tmp/model_' + folder + '.ckpt')
			print("Model saved in file: %s" % save_path)
