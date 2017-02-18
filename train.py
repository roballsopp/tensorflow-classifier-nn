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
	np.random.shuffle(data)

	np_x_train = data['X'][:-1000]
	np_y_train = data['y'][:-1000]
	np_x_val = data['X'][-1000:]
	np_y_val = data['y'][-1000:]

	graph = tf.Graph()
	with graph.as_default():
		x_train_init = tf.placeholder(dtype=tf.float32, shape=np_x_train.shape)
		y_train_init = tf.placeholder(dtype=tf.uint8, shape=np_y_train.shape)
		x_val_init = tf.placeholder(dtype=tf.float32, shape=np_x_val.shape)
		y_val_init = tf.placeholder(dtype=tf.uint8, shape=np_y_val.shape)

		x_train = tf.Variable(x_train_init, trainable=False)
		y_train = tf.Variable(y_train_init, trainable=False)
		x_val = tf.Variable(x_val_init, trainable=False)
		y_val = tf.Variable(y_val_init, trainable=False)

		net = nn.FullyConnected(layers)

		hyp_train = net.forward_prop(x_train)
		hyp_val = net.forward_prop(x_val)

		cost_train = nn.cross_entropy(hyp_train, y_train)
		cost_val = nn.cross_entropy(hyp_val, y_val)

		optimize = tf.train.AdamOptimizer().minimize(cost_train)

		metrics_train = nn.evaluate(hyp_train, y_train)
		metrics_val = nn.evaluate(hyp_val, y_val)

		summaries_train = eval_train(metrics_train, cost_train)
		summaries_val = eval_val(metrics_val, cost_val)

		init = tf.global_variables_initializer()

		with tf.Session() as sess:
			session_saver = net.get_saver()
			summary_writer = tf.summary.FileWriter('./tmp/logs/' + folder)
			sess.run([
				x_train.initializer,
				y_train.initializer,
				x_val.initializer,
				y_val.initializer,
				init
			], feed_dict={
				x_train_init: np_x_train,
				y_train_init: np_y_train,
				x_val_init: np_x_val,
				y_val_init: np_y_val
			})

			NUM_STEPS = 4000

			def add_summary(step):
				train_results, val_results = sess.run([summaries_train, summaries_val])
				summary_writer.add_summary(train_results, step)
				summary_writer.add_summary(val_results, step)
				print('Step', step + 1, 'of', NUM_STEPS)

			for step in range(NUM_STEPS):
				sess.run(optimize)
				every_n_steps(10, step, add_summary)

			save_path = session_saver.save(sess, './tmp/model_' + folder + '.ckpt')
			print("Model saved in file: %s" % save_path)
