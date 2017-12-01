import logging
import time
import os
from argparse import ArgumentParser

import tensorflow as tf

from load import Wave
from predict.model import Model

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

def normalize_out(signal):
	sig_max = tf.reduce_max(tf.abs(signal))
	return signal / sig_max

inputs = wav.get_data()
inputs = tf.convert_to_tensor(inputs[:, :500000], dtype=tf.float32)

model = Model(inputs)
hypothesis = model.forward_prop()

init = tf.global_variables_initializer()

with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
	sess.run(init)
	output = sess.run(normalize_out(hypothesis))

	wav_out = Wave(output, sample_rate=wav.sample_rate)
	wav_out.to_file(os.path.join(job_dir, job_name + '_out.wav'))
