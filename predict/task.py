import logging
import time
import os
from argparse import ArgumentParser

import tensorflow as tf
import nn

from load import Wave
from predict.model import Model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

parser = ArgumentParser()

parser.add_argument('--input-file', help="Specify input file path", required=True)
parser.add_argument('--label-file', help="Specify label file path")
parser.add_argument('--job-dir', type=str, default='./tmp/wav_out')
parser.add_argument('--job-name', type=str, default=time.strftime('%Y-%m-%d_%H-%M-%S'))

args = parser.parse_args()

input_filepath = args.input_file
label_filepath = args.label_file
job_dir = args.job_dir
job_name = args.job_name

logging.info('Loading input file ' + input_filepath + '...')
wav = Wave.from_file(input_filepath)

labels = Wave.from_file(label_filepath)
labels = labels.get_data()
labels = tf.convert_to_tensor(labels, dtype=tf.float32)

inputs = wav.get_data()
inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)

inputs = tf.transpose(inputs)[:500000, :]
labels = tf.transpose(labels)[:500000, :]

model = Model(inputs, channels_last=True)
predictions = model.forward_prop()
cost = Model.cost(predictions, labels)

init = tf.global_variables_initializer()

with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
	sess.run(init)
	cost, output = sess.run([cost, tf.transpose(nn.normalize(predictions))])

	print('Total Cost: ', cost)

	wav_out = Wave(output, sample_rate=wav.sample_rate)
	wav_out.to_file(os.path.join(job_dir, job_name + '_out.wav'))
