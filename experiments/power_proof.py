import logging
import time
from argparse import ArgumentParser

import tensorflow as tf
import nn
from matplotlib import pyplot as plt
import numpy as np

from load import Wave, WaveTF

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
wav = WaveTF.from_file(input_filepath)

inputs = wav.get_data()

inputs = inputs[:, 6264000:6276000]

fft_length = 512

input_first_frame = inputs[:, :fft_length]

# IMPORTANT: don't forget the effects of the window fn. stft adds window automatically, and power measurements wont be accurate using this method
fft = tf.spectral.rfft(input_first_frame, fft_length=[fft_length])
mag_first_frame = tf.abs(fft)

print(mag_first_frame.get_shape().as_list())

init = tf.global_variables_initializer()

# interesting that its the mean for fft
first_frame_power_stft = tf.reduce_mean(mag_first_frame ** 2)
first_frame_power_input = tf.reduce_sum(input_first_frame ** 2)

with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
	sess.run(init)
	out = sess.run([first_frame_power_stft, first_frame_power_input])
	print(out)

