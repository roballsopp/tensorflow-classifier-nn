import logging
import time
from argparse import ArgumentParser

import tensorflow as tf
import nn
from matplotlib import pyplot as plt
import numpy as np

from load import Wave

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

parser = ArgumentParser()

parser.add_argument('--input-file', help="Specify input file path", required=True)
parser.add_argument('--job-dir', type=str, default='./tmp/wav_out')
parser.add_argument('--job-name', type=str, default=time.strftime('%Y-%m-%d_%H-%M-%S'))

args = parser.parse_args()

input_filepath = args.input_file
job_dir = args.job_dir
job_name = args.job_name

logging.info('Loading input file ' + input_filepath + '...')
wav = Wave.from_file(input_filepath)

inputs = wav.get_data()
inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)

inputs = inputs[0, 6264000:6276000]

def diff(x, n=1):
	diffed = tf.pad(x[1:] - x[:-1], [[1, 0]])

	if n > 1:
		return diff(diffed, n=n-1)

	return diffed

diffed = diff(inputs, n=1)
signs = nn.math.fix_divide_by_zero(diffed / tf.abs(diffed))
crossings = tf.abs(diff(signs, n=1))

avg_crossings = tf.layers.average_pooling1d(
	tf.expand_dims(tf.expand_dims(crossings, axis=0), axis=2),
	pool_size=256,
	strides=1,
	padding='same',
	data_format='channels_last'
)

avg_crossings = tf.squeeze(avg_crossings, axis=[0, 2])

init = tf.global_variables_initializer()

with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
	sess.run(init)
	inputs, signs = sess.run([inputs, avg_crossings])

	print(signs)

	num_plots = 2

	plt.figure(1)

	plt.subplot(num_plots, 1, 1)
	plt.axis([0, len(inputs), np.min(inputs), np.max(inputs)])
	plt.plot(inputs)

	plt.subplot(num_plots, 1, 2)
	plt.axis([0, len(signs), np.min(signs), np.max(signs)])
	plt.plot(signs)

	plt.show()
