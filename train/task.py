import tensorflow as tf
import time
import logging
import os
from argparse import ArgumentParser
import nn
import loadData

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

parser = ArgumentParser()

parser.add_argument('input_pattern', help="Specify input file(s) using glob pattern")
parser.add_argument('layers',
										help="Specify nn hidden layer architecture. Provide space separated integers to specify the number of neurons in each hidden layer.",
										nargs='*',
										type=int)
parser.add_argument('-n', '--num-iters', type=int, help="Specify number of training iterations.", default=4000)
parser.add_argument('-s', '--start-iter', type=int, help="Specify the training iteration to start on.", default=0)
parser.add_argument('-c', '--from-checkpoint', type=str, help="Specify a checkpoint to restore from.")

def every_n_steps(n, step, callback):
	if (step > 0) and ((step + 1) % n == 0):
		callback(step)

class TrainerGraph:
	def __init__(self, net, x_shape, y_shape, x_type=tf.float32, y_type=tf.float32):
		self.x_init = tf.placeholder(dtype=x_type, shape=x_shape)
		self.y_init = tf.placeholder(dtype=y_type, shape=y_shape)

		self.x = tf.Variable(self.x_init, trainable=False)
		self.y = tf.Variable(self.y_init, trainable=False)

		self.hyp = net.forward_prop(self.x)
		self.cost = nn.cross_entropy(self.hyp, self.y)

	def evaluate(self, summary_namespace):
		metrics = nn.evaluate(self.hyp, self.y)
		summaries = [tf.summary.scalar('cost_' + summary_namespace, self.cost)]

		for metric_name in metrics:
			summary = tf.summary.scalar(metric_name + '_' + summary_namespace, metrics[metric_name])
			summaries.append(summary)

		return tf.summary.merge(summaries)


def train(layers, dataset, num_steps=4000, restore_variables_from="", step_start=0):
	run_name = '_'.join(map(str, layers)) + ' - ' + time.strftime('%Y-%m-%d_%H-%M-%S')
	dataset.shuffle()

	x_train = dataset.features[:-1000]
	y_train = dataset.labels[:-1000]
	x_val = dataset.features[-1000:]
	y_val = dataset.labels[-1000:]

	graph = tf.Graph()
	with graph.as_default():
		net = nn.FullyConnected(layers)

		graph_train = TrainerGraph(net, x_train.shape, y_train.shape, x_train.dtype, y_train.dtype)
		graph_val = TrainerGraph(net, x_val.shape, y_val.shape, x_val.dtype, y_val.dtype)

		optimize = tf.train.AdamOptimizer().minimize(graph_train.cost)

		summaries_train = graph_train.evaluate('train')
		summaries_val = graph_val.evaluate('val')

		init = tf.global_variables_initializer()

		with tf.Session() as sess:
			session_saver = net.get_saver()
			summary_writer = tf.summary.FileWriter(os.path.join('./tmp', run_name), graph=sess.graph)

			sess.run(init, feed_dict={
				graph_train.x_init: x_train,
				graph_train.y_init: y_train,
				graph_val.x_init: x_val,
				graph_val.y_init: y_val
			})

			# must come after sess.run(init) or the restored vars will be wiped out
			if restore_variables_from != "":
				session_saver.restore(sess, restore_variables_from)

			def add_summary(step):
				train_results, val_results = sess.run([summaries_train, summaries_val])
				summary_writer.add_summary(train_results, step)
				summary_writer.add_summary(val_results, step)
				logging.info('Step ' + str(step + 1) + ' of ' + str(num_steps))

			def save_checkpoint(step=None, name='checkpoint', write_meta_graph=False):
				save_path = os.path.join('./tmp', run_name, name)
				save_path = session_saver.save(sess, save_path, global_step=step, write_meta_graph=write_meta_graph)
				logging.info("Model saved at: %s" % save_path)

			for step in range(step_start, num_steps):
				sess.run(optimize)
				every_n_steps(10, step, add_summary)
				every_n_steps(100, step, save_checkpoint)

			save_checkpoint(name='export', write_meta_graph=True)


args = parser.parse_args()

logging.info('Loading files matching ' + args.input_pattern + '...')
dataset = loadData.load(args.input_pattern)
logging.info('Files loaded successfully. Loaded ' + str(dataset.num_examples) + ' training examples')

layers = [dataset.num_features] + args.layers + [dataset.num_labels]
logging.info('Training neural network with architecture ' + ', '.join(map(str, layers)) + '...')

train(layers, dataset, num_steps=args.num_iters, restore_variables_from=args.from_checkpoint, step_start=args.start_iter)
