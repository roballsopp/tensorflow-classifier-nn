import tensorflow as tf
import time
import logging
import os
import glob
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
parser.add_argument('-n', '--num-iters', type=int, help="Specify number of training iterations.", default=100000)
parser.add_argument('-s', '--start-iter', type=int, help="Specify the training iteration to start on.", default=0)
parser.add_argument('-c', '--from-checkpoint', type=str, help="Specify a checkpoint to restore from.")

def every_n_steps(n, step, callback):
	if (step > 0) and ((step + 1) % n == 0):
		callback(step)

class TrainerGraph:
	def __init__(self, inputs, labels, reuse=None):
		self.y = labels

		net = nn.Net(inputs, reuse)

		self.hyp = net.forward_prop()
		self.cost = net.loss(labels)

	def evaluate(self, summary_namespace):
		metrics = nn.evaluate(self.hyp, self.y)
		summaries = [tf.summary.scalar('cost_' + summary_namespace, self.cost)]

		for metric_name in metrics:
			summary = tf.summary.scalar(metric_name + '_' + summary_namespace, metrics[metric_name])
			summaries.append(summary)

		return tf.summary.merge(summaries)


def train(hidden_layers, train_dataset, val_dataset, num_steps=100000, restore_variables_from=None, step_start=0):
	run_name = '_'.join(map(str, layers)) + ' - ' + time.strftime('%Y-%m-%d_%H-%M-%S')

	val_dataset = val_dataset.cache().repeat().batch(1000)
	train_dataset = train_dataset.repeat().batch(1000)

	iter_data_train = train_dataset.make_initializable_iterator()
	iter_data_val = val_dataset.make_initializable_iterator()

	x_train, y_train = iter_data_train.get_next()
	x_val, y_val = iter_data_val.get_next()

	graph_train = TrainerGraph(x_train, y_train)
	graph_val = TrainerGraph(x_val, y_val, reuse=True)

	optimize = tf.train.AdamOptimizer().minimize(graph_train.cost)

	summaries_train = graph_train.evaluate('train')
	summaries_val = graph_val.evaluate('val')

	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		session_saver = tf.train.Saver()
		summary_writer = tf.summary.FileWriter(os.path.join('./tmp', run_name), graph=sess.graph)

		sess.run([init, iter_data_train.initializer, iter_data_val.initializer])

		# must come after sess.run(init) or the restored vars will be wiped out
		if restore_variables_from:
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
			every_n_steps(100, step, add_summary)
			every_n_steps(1000, step, save_checkpoint)

		save_checkpoint(name='export', write_meta_graph=True)


args = parser.parse_args()

logging.info('Loading files matching ' + args.input_pattern + '...')
filenames = glob.glob(args.input_pattern, recursive=True)
logging.info('Found ' + str(len(filenames)) + ' files.')

train_dataset = loadData.load(filenames[1:])
val_dataset = loadData.load(filenames[0:1])

logging.info('Files loaded successfully.')

features_shape, labels_shape = train_dataset.output_shapes

num_features = features_shape.as_list()[0]
num_labels = labels_shape.as_list()[0]

layers = [num_features] + args.layers + [num_labels]
logging.info('Training neural network with architecture ' + ', '.join(map(str, layers)) + '...')

train(args.layers, train_dataset, val_dataset, num_steps=args.num_iters, restore_variables_from=args.from_checkpoint, step_start=args.start_iter)
