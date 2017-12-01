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
hypothesis = model.get_raw()

init = tf.global_variables_initializer()

with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
	sess.run(init)
	output = sess.run(normalize_out(hypothesis))

	wav_out = Wave(output, sample_rate=wav.sample_rate)
	wav_out.to_file(os.path.join(job_dir, job_name + '_out.wav'))


# take a plain array (x) and convert it into slices of size num_features, possibly overlapping them
# def build_inputs(x, num_features, overlap):
# 	x_length = x.shape.as_list()[0]
# 	num_features = tf.constant(num_features, name="num_features")
# 	overlap = tf.constant(overlap, name="overlap")
# 	base_size = num_features - overlap
# 	remainder = x_length % base_size
#
# 	def pad_input():
# 		input_padding = tf.zeros([base_size - remainder], name="input_padding")
# 		return tf.concat([x, input_padding], 0, name="append_input_padding")
#
# 	x = tf.cond(remainder > 0, pad_input, lambda: x)
# 	x = tf.reshape(x, [-1, base_size])
#
# 	overlap_rows = x[1:, 0:overlap]
# 	overlap_padding = tf.zeros([1, overlap], name="overlap_padding")
# 	overlap_rows = tf.concat([overlap_rows, overlap_padding], 0, name="append_overlap_padding")
#
# 	return tf.concat([x, overlap_rows], 1, name="create_final_inputs")

def upsample(value, output_length):
	ratio = output_length / value.shape.as_list()[0]
	indicies_first_dim = tf.zeros(value.shape, dtype=tf.int64)
	indicies_second_dim = tf.to_int64(tf.round(tf.range(0., output_length, ratio)))
	indicies = tf.stack((indicies_first_dim, indicies_second_dim), axis=1)
	input = tf.SparseTensor(indices=indicies, values=value, dense_shape=(1, output_length))
	return tf.sparse_tensor_to_dense(input)
