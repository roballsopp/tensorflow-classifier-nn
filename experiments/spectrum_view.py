import logging
import time
from argparse import ArgumentParser

import tensorflow as tf
import nn
from matplotlib import pyplot as plt
import numpy as np

from load import WaveTF
from scipy.signal import get_window

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def build_fft_window(name, length, padded_len=None):
	padded_len = padded_len if padded_len is not None else length

	fft_window = tf.convert_to_tensor(get_window(name, length), dtype=tf.float32)
	# make it the same length as the signal
	fft_window = tf.pad(fft_window, [[0, padded_len - length]])
	# add channel dim to window
	fft_window = tf.expand_dims(fft_window, axis=0)
	return fft_window

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

audio_data = wav.get_data()
sr = wav.sample_rate

# linkin park start: 24, len: 10, no vox
# veil of maya start: 2, len: 10, vox
# awaiting winter: start: 10, len: 10, no nox
start = sr * 10
length = (sr * 10)
end = start + length

next_power_2 = int(np.ceil(np.log2(length)))
fft_length = np.power(2, next_power_2)
print('fft length:', fft_length)
print('fft/sig ratio:', fft_length / length)

fft_inputs = tf.pad(audio_data[:, start:end], [[0, 0], [0, fft_length - length]])

# IMPORTANT: don't forget the effects of the window fn.
# fft_window = build_fft_window('blackman', length=sr, padded_len=length)

fft = tf.spectral.rfft(fft_inputs, fft_length=[fft_length])
mag = tf.abs(fft) / fft_length

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	inputs, mag = sess.run([fft_inputs, mag])

	num_plots = 2

	plt.figure(1)

	plt.subplot(num_plots, 1, 1)
	for chan in inputs:
		plt.plot(chan, alpha=0.8)

	plt.subplot(num_plots, 1, 2)
	# keep the x axis always on a scale from 0 to sr / 2
	sr2 = sr / 2
	step = sr2 / len(mag[0])
	x_values = np.arange(sr2, step=step)
	for chan in mag:
		mag_db = 20 * np.log10(chan)
		plt.plot(x_values, mag_db, alpha=0.8)
	plt.xscale('log')  # psychoacoustic scaling of frequency bins
	plt.ylim([-150, 0])  # we converted mag to db, so max db should be 0
	plt.xlim([0.1, sr])
	plt.grid()

	plt.show()
