import tensorflow as tf
import time
import logging
from argparse import ArgumentParser
import nn
import loadData

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

parser = ArgumentParser()

parser.add_argument('input_pattern', help="Specify input file(s) using glob pattern")
parser.add_argument('layers',
										help="Specify nn hidden layer architecture. Provide space separated integers to specify the number of neurons in heach hidden layer.",
										nargs='*',
										type=int)
parser.add_argument('-s', '--steps', type=int, help="Specify number of training iterations.", default=4000)

def every_n_steps(n, step, callback):
	if (step > 0) and ((step + 1) % n == 0):
		callback(step)

class TrainerGraph:
	def __init__(self, net, x_shape, y_shape):
		self.x_init = tf.placeholder(dtype=tf.float32, shape=x_shape)
		self.y_init = tf.placeholder(dtype=tf.uint8, shape=y_shape)

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


def train(layers, dataset, num_steps=4000):
	run_name = '_'.join(map(str, layers)) + ' - ' + time.strftime('%Y-%m-%w_%H-%M-%S')
	dataset.shuffle()

	x_train = dataset.features[:-1000]
	y_train = dataset.labels[:-1000]
	x_val = dataset.features[-1000:]
	y_val = dataset.labels[-1000:]

	graph = tf.Graph()
	with graph.as_default():
		net = nn.FullyConnected(layers)

		graph_train = TrainerGraph(net, x_train.shape, y_train.shape)
		graph_val = TrainerGraph(net, x_val.shape, y_val.shape)

		optimize = tf.train.AdamOptimizer().minimize(graph_train.cost)

		summaries_train = graph_train.evaluate('train')
		summaries_val = graph_val.evaluate('val')

		init = tf.global_variables_initializer()

		with tf.Session() as sess:
			session_saver = net.get_saver()
			summary_writer = tf.summary.FileWriter('./tmp/' + run_name)
			sess.run(init, feed_dict={
				graph_train.x_init: x_train,
				graph_train.y_init: y_train,
				graph_val.x_init: x_val,
				graph_val.y_init: y_val
			})

			def add_summary(step):
				train_results, val_results = sess.run([summaries_train, summaries_val])
				summary_writer.add_summary(train_results, step)
				summary_writer.add_summary(val_results, step)
				logging.info('Step ' + str(step + 1) + ' of ' + str(num_steps))

			def save_model(step=None):
				save_path = session_saver.save(sess, './tmp/' + run_name + '/model', global_step=step)
				logging.info("Model saved at: %s" % save_path)

			for step in range(num_steps):
				sess.run(optimize)
				every_n_steps(10, step, add_summary)
				every_n_steps(100, step, save_model)

			save_model()

args = parser.parse_args()

logging.info('Loading files matching ' + args.input_pattern + '...')
dataset = loadData.load(args.input_pattern)
logging.info('Files loaded successfully. Loaded ' + str(dataset.num_examples) + ' training examples')

layers = [dataset.num_features] + args.layers + [dataset.num_labels]
logging.info('Training neural network with architecture ' + ', '.join(map(str, layers)) + '...')

train(layers, dataset, num_steps=args.steps)
