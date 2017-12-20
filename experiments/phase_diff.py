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

fft_length = 512
fft2 = int(fft_length / 2)

# IMPORTANT: don't forget the effects of the window fn. stft adds window automatically, and power measurements wont be accurate using this method
stft_inputs = tf.pad(inputs, [[fft2, fft2]])
stft = tf.contrib.signal.stft(stft_inputs, frame_length=fft_length, frame_step=1, fft_length=fft_length)
mag = tf.transpose(tf.abs(stft))
phase = tf.transpose(tf.angle(stft))

def tf_unwrap(p):
	diff = p[1:, :] - p[:-1, :]
	diff_mod = (diff + np.pi) % (2 * np.pi) - np.pi
	cond = (diff_mod == -np.pi) & (diff > 0)
	diff_mod = tf.where(cond, tf.constant(np.pi, shape=diff.shape), diff_mod)
	ph_correct = diff_mod - diff
	ph_correct = tf.where(tf.abs(diff) < np.pi, tf.zeros(diff.shape), ph_correct)
	return p + tf.pad(tf.cumsum(ph_correct, axis=0), [[1, 0], [0, 0]])


phase = tf_unwrap(phase)

mag_log = tf.log(mag + 1e-16)
mag_log = nn.normalize(mag_log - tf.reduce_min(mag_log))

# the idea here was that there might be some crazy phase changes happening during quite parts of the signal that
# we don't care about, so go ahead and ignore values that happen during quite parts. plug adjusted_phase into the
# reduce_sum calculations below to see it work. also, mag_log makes everything seem pretty loud, try mag instead
adjusted_phase = mag_log * phase

mag_sum = tf.reduce_sum(mag, axis=[0])
mag_diff = tf.pad(mag_sum[1:] - mag_sum[:-1], [[1, 0]])

phase_sum = tf.reduce_sum(phase, axis=[0])

mag_norm = nn.rms_normalize(mag_diff)
phase_norm = nn.rms_normalize(phase_sum)

mag_phase_sum = tf.nn.relu(mag_norm + phase_norm)
# mag_phase_f1 is supposed to be a compromise between phase adjusted by magnitude, and phase and magnitude together
# to see this all the way through id probably want to adjust the phase by magnitude before summing each frequency band
mag_phase_f1 = ((tf.nn.relu(mag_norm) * tf.nn.relu(phase_norm)) + mag_phase_sum) / 2

# make magnitude a bit more visible
mag = tf.pow(nn.normalize(mag), 0.3)

init = tf.global_variables_initializer()

with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
	sess.run(init)
	inputs, mag, mag_log, phase, mag_sum, phase_sum, mag_phase_sum, adjusted_phase, mag_phase_f1 = sess.run([
		inputs,
		mag,
		mag_log,
		phase,
		mag_diff,
		phase_sum,
		mag_phase_sum,
		adjusted_phase,
		mag_phase_f1
	])

	num_plots = 8

	plt.figure(1)

	plt.subplot(num_plots, 1, 1)
	plt.axis([0, len(inputs), np.min(inputs), np.max(inputs)])
	plt.plot(inputs)

	plt.subplot(num_plots, 1, 2)
	plt.imshow(mag)

	plt.subplot(num_plots, 1, 3)
	plt.imshow(phase)

	plt.subplot(num_plots, 1, 4)
	plt.imshow(adjusted_phase)

	plt.subplot(num_plots, 1, 5)
	plt.axis([0, len(mag_sum), np.min(mag_sum), np.max(mag_sum)])
	plt.plot(mag_sum)

	plt.subplot(num_plots, 1, 6)
	plt.axis([0, len(phase_sum), np.min(phase_sum), np.max(phase_sum)])
	plt.plot(phase_sum)

	plt.subplot(num_plots, 1, 7)
	plt.axis([0, len(mag_phase_sum), np.min(mag_phase_sum), np.max(mag_phase_sum)])
	plt.plot(mag_phase_sum)

	plt.subplot(num_plots, 1, 8)
	plt.axis([0, len(mag_phase_f1), np.min(mag_phase_f1), np.max(mag_phase_f1)])
	plt.plot(mag_phase_f1)

	plt.show()
