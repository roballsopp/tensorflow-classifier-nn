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

signal = inputs[0, 6264000:6276000]
sig_rect = tf.abs(signal)

conv_input = tf.expand_dims(sig_rect, axis=0)  # batch
conv_input = tf.expand_dims(conv_input, axis=2)  # height
conv_input = tf.expand_dims(conv_input, axis=3)  # chan

sig_diff = tf.nn.convolution(
	conv_input,
	filter=nn.transient_kernel([512, 1, 1, 1]),
	padding='SAME',
	data_format='NHWC'
)

sig_diff = tf.squeeze(sig_diff)
sig_diff = tf.nn.relu(sig_diff)
# sig_diff = nn.smooth(sig_diff, 64)
sig_diff = tf.log(nn.normalize(sig_diff) + 1e-16)

init = tf.global_variables_initializer()

with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
	sess.run(init)
	signal, sig_rect, sig_diff = sess.run([signal, sig_rect, sig_diff])

	num_plots = 3

	plt.figure(1)

	plt.subplot(num_plots, 1, 1)
	plt.axis([0, len(signal), np.min(signal), np.max(signal)])
	plt.plot(signal)

	plt.subplot(num_plots, 1, 2)
	plt.axis([0, len(sig_rect), np.min(sig_rect), np.max(sig_rect)])
	plt.plot(sig_rect)

	plt.subplot(num_plots, 1, 3)
	plt.axis([0, len(sig_diff), -5, np.max(sig_diff)])
	plt.plot(sig_diff)

	plt.show()
