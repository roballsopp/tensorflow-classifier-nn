import logging
import os
import time
import glob
from argparse import ArgumentParser

import tensorflow as tf

import load
from train.model import Model
from train.eval import evaluate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

parser = ArgumentParser()

parser.add_argument('--train-files', help="Specify training data file(s) using glob pattern")
parser.add_argument('--eval-files', help="Specify evaluation data file(s) using glob pattern")
parser.add_argument('--file-bucket', help="Specify gcloud bucket to use for data files")
parser.add_argument('--job-dir', type=str, default='./tmp')
parser.add_argument('--job-name', type=str, default=time.strftime('%Y-%m-%d_%H-%M-%S'))
parser.add_argument('-n', '--num-iters', type=int, help="Specify number of training iterations.", default=100000)
parser.add_argument('-s', '--start-iter', type=int, help="Specify the training iteration to start on.", default=0)
parser.add_argument('-c', '--from-checkpoint', type=str, help="Specify a checkpoint to restore from.")

def every_n_steps(n, step, callback):
	if (step > 0) and ((step + 1) % n == 0):
		callback(step)

class TrainerGraph:
	def __init__(self, inputs, labels, reuse=None):
		self.y = labels

		net = Model(inputs, reuse)

		self.hyp = net.forward_prop()
		self.cost = net.loss(labels)

	def evaluate(self, summary_namespace):
		metrics = evaluate(self.hyp, self.y)
		summaries = [tf.summary.scalar('cost_' + summary_namespace, self.cost)]

		for metric_name in metrics:
			summary = tf.summary.scalar(metric_name + '_' + summary_namespace, metrics[metric_name])
			summaries.append(summary)

		return tf.summary.merge(summaries)


args = parser.parse_args()

job_dir = args.job_dir
job_name = args.job_name
restore_variables_from = args.from_checkpoint
step_start = args.start_iter
num_steps = args.num_iters

train_dataset = None
val_dataset = None

if args.file_bucket:
	logging.info('Loading files from bucket ' + args.file_bucket + '...')

	train_dataset = load.from_bucket(args.file_bucket, 'train')
	val_dataset = load.from_bucket(args.file_bucket, 'eval')
else:
	logging.info('Loading training files matching ' + args.train_files + '...')
	training_filenames = glob.glob(args.train_files, recursive=True)
	logging.info('Found ' + str(len(training_filenames)) + ' files.')

	logging.info('Loading evaluation files matching ' + args.eval_files + '...')
	eval_filenames = glob.glob(args.eval_files, recursive=True)
	logging.info('Found ' + str(len(eval_filenames)) + ' files.')

	train_dataset = load.from_filenames(training_filenames)
	val_dataset = load.from_filenames(eval_filenames)

logging.info('Building neural network...')

val_dataset = val_dataset.shuffle(10000).repeat().batch(256)
train_dataset = train_dataset.shuffle(10000).repeat().batch(256)

iter_data_train = train_dataset.make_initializable_iterator()
iter_data_val = val_dataset.make_initializable_iterator()

x_train, y_train = iter_data_train.get_next()
x_val, y_val = iter_data_val.get_next()

graph_train = TrainerGraph(x_train, y_train)
graph_val = TrainerGraph(x_val, y_val, reuse=True)

optimize = tf.train.AdamOptimizer(learning_rate=0.001).minimize(graph_train.cost)

summaries_train = graph_train.evaluate('train')
summaries_val = graph_val.evaluate('val')

init = tf.global_variables_initializer()

with tf.Session() as sess:
	session_saver = tf.train.Saver()
	summary_writer = tf.summary.FileWriter(os.path.join(job_dir, job_name), graph=sess.graph)

	sess.run([init, iter_data_train.initializer, iter_data_val.initializer])

	# must come after sess.run(init) or the restored vars will be wiped out
	if restore_variables_from:
		session_saver.restore(sess, restore_variables_from)

	def add_summary(step):
		# audio_summary = tf.summary.audio('audio_data', x_train, 11025)
		# label_summary = tf.summary.audio('label_data', y_train, 11025)
		# audio_results, label_results = sess.run([audio_summary, label_summary])
		# summary_writer.add_summary(audio_results, step)
		# summary_writer.add_summary(label_results, step)

		train_results, val_results = sess.run([summaries_train, summaries_val])
		summary_writer.add_summary(train_results, step)
		summary_writer.add_summary(val_results, step)
		logging.info('Step ' + str(step + 1) + ' of ' + str(num_steps))

	def save_checkpoint(step=None, name='checkpoint', write_meta_graph=False):
		save_path = os.path.join(job_dir, job_name, name)
		save_path = session_saver.save(sess, save_path, global_step=step, write_meta_graph=write_meta_graph)
		logging.info("Model saved at: %s" % save_path)

	logging.info('Training neural network...')

	for step in range(step_start, num_steps):
		sess.run(optimize)
		every_n_steps(10, step, add_summary)
		every_n_steps(100, step, save_checkpoint)

	save_checkpoint(name='export', write_meta_graph=True)
