import logging
import os
import time
import glob
from argparse import ArgumentParser

import tensorflow as tf

import load
from train.dual_model import Model
from train.eval import evaluate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

parser = ArgumentParser()

parser.add_argument('--train-files', help="Specify training data file(s) using glob pattern")
parser.add_argument('--eval-files', help="Specify evaluation data file(s) using glob pattern")
parser.add_argument('--file-bucket', help="Specify gcloud bucket to use for data files")
parser.add_argument('--job-dir', type=str, default='./tmp')
parser.add_argument('--job-name', type=str, default=time.strftime('%Y-%m-%d_%H-%M-%S'))
parser.add_argument('--num-iters', type=int, help="Specify number of training iterations.", default=100000)
parser.add_argument('--start-iter', type=int, help="Specify the training iteration to start on.", default=0)
parser.add_argument('--learning-rate', type=float, help="Specify the learning rate.", default=0.001)
parser.add_argument('--train-batch-size', type=int, help="Specify the training batch size.", default=256)
parser.add_argument('--log-device-placement', type=bool, help="Log the device eah op is placed on.", default=False)
parser.add_argument('--channels-last', type=bool, help="Specify the data format to build the model with", default=False)
parser.add_argument('--from-checkpoint', type=str, help="Specify a checkpoint to restore from.")


def every_n_steps(n, step, callback):
	if (step > 0) and ((step + 1) % n == 0):
		callback(step)


def build_summaries(metrics, family=None):
	summaries = []

	for metric_name in metrics:
		summary = tf.summary.scalar(metric_name, metrics[metric_name], family=family)
		summaries.append(summary)

	return tf.summary.merge(summaries)

args = parser.parse_args()

job_dir = args.job_dir
restore_variables_from = args.from_checkpoint
step_start = args.start_iter
num_steps = args.num_iters
learning_rate = args.learning_rate
train_batch_size = args.train_batch_size
channels_last = args.channels_last
log_device_placement = args.log_device_placement
job_name = args.job_name + ' - LR-' + str(learning_rate) + ' - B-' + str(train_batch_size)

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

	train_dataset = load.from_filenames(training_filenames, channels_last)
	val_dataset = load.from_filenames(eval_filenames, channels_last)

logging.info('Building neural network...')

val_dataset = val_dataset.repeat().batch(256)
train_dataset = train_dataset.repeat().shuffle(40000).batch(train_batch_size)

iter_data_train = train_dataset.make_initializable_iterator()
iter_data_val = val_dataset.make_initializable_iterator()

batch_train = iter_data_train.get_next()
batch_eval = iter_data_val.get_next()

use_eval = tf.placeholder(tf.bool, name='use_eval')

time_series_features, labels = tf.cond(use_eval, lambda: batch_eval, lambda: batch_train)

model = Model(time_series_features, training=tf.logical_not(use_eval), reuse=False, channels_last=channels_last)
hypothesis = model.forward_prop()
cost = model.loss(labels)

global_step = tf.Variable(step_start, trainable=False, name='global_step')

learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps=10000, decay_rate=0.5, staircase=True)

optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

metrics = evaluate(hypothesis, labels)
metrics['cost'] = cost

train_summaries = build_summaries(metrics, family='train')
eval_summaries = build_summaries(metrics, family='eval')
lr_summary = tf.summary.scalar('learning_rate', learning_rate)

summaries = tf.cond(use_eval, lambda: eval_summaries, lambda: train_summaries)
summaries = tf.summary.merge([summaries, lr_summary])

init = tf.global_variables_initializer()

with tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement)) as sess:
	model_vars = model.get_savable_vars()
	session_saver = tf.train.Saver(var_list=model_vars)
	summary_writer = tf.summary.FileWriter(os.path.join(job_dir, job_name), graph=sess.graph)

	sess.run([init, iter_data_train.initializer, iter_data_val.initializer])

	# must come after sess.run(init) or the restored vars will be wiped out
	if restore_variables_from:
		session_saver.restore(sess, restore_variables_from)

	def add_summary(step):
		# spectrogram_summary = tf.summary.image('spectrogram_data', tf.reshape(spectrogram_features, [128, 33, 256, 1]), max_outputs=50)
		# time_series_summary = tf.summary.audio('time_series_data', tf.reshape(time_series_features, [128, 256, 1]), 11025, max_outputs=50)
		# summary_buf = sess.run(tf.summary.merge([spectrogram_summary, time_series_summary]), feed_dict={use_eval: False})
		# summary_writer.add_summary(summary_buf, step)

		train_summary_buf = sess.run(summaries, feed_dict={use_eval: False})
		eval_summary_buf = sess.run(summaries, feed_dict={use_eval: True})

		summary_writer.add_summary(train_summary_buf, step)
		summary_writer.add_summary(eval_summary_buf, step)
		logging.info('Step ' + str(step + 1) + ' of ' + str(num_steps))

	def save_checkpoint(step=None, name='checkpoint', write_meta_graph=False):
		save_path = os.path.join(job_dir, job_name, name)
		save_path = session_saver.save(sess, save_path, global_step=step, write_meta_graph=write_meta_graph)
		logging.info("Model saved at: %s" % save_path)

	logging.info('Training neural network...')

	for step in range(step_start, num_steps):
		sess.run(optimize, feed_dict={use_eval: False})
		every_n_steps(10, step, add_summary)
		every_n_steps(100, step, save_checkpoint)

	save_checkpoint(name='export', write_meta_graph=True)

