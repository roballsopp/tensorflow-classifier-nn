import logging
import time

import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import get_window

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def get_mag(x, fft_length=512):
	window_len = x.shape[-1].value
	W1 = int(round(window_len / 2))
	W2 = window_len - W1
	padding = tf.zeros([fft_length - window_len], dtype=tf.float32)
	#  zero phase windowing
	x = tf.concat([x[W2:], padding, x[:W1]], axis=0)
	fft = tf.spectral.rfft(x, fft_length=[fft_length])
	mag = tf.abs(fft)
	return tf.concat([tf.reverse(mag[1:], axis=[0]), mag], axis=0)

window_length = 511

rectangular = tf.ones([window_length], dtype=tf.float32)
hamming = tf.contrib.signal.hamming_window(window_length)
hann = tf.contrib.signal.hann_window(window_length)
hanning = tf.convert_to_tensor(get_window('hanning', window_length), dtype=tf.float32)
blackman = tf.convert_to_tensor(get_window('blackman', window_length), dtype=tf.float32)

fft_length = 512

mag_rect = get_mag(rectangular, fft_length=fft_length)
mag_ham = get_mag(hamming, fft_length=fft_length)
mag_hann = get_mag(hann, fft_length=fft_length)
mag_hanning = get_mag(hanning, fft_length=fft_length)
mag_blackman = get_mag(blackman, fft_length=fft_length)

init = tf.global_variables_initializer()

def log10(x):
	numerator = tf.log(x)
	denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
	return numerator / denominator

with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
	sess.run(init)
	out = np.array(sess.run([mag_rect, mag_ham, mag_hann, mag_hanning, mag_blackman]))

	mag_rect, mag_ham, mag_hann, mag_hanning, mag_blackman = 20 * np.log10(out + 1e-16)

	num_plots = 5

	plt.figure(1)

	plt.subplot(num_plots, 1, 1)
	plt.plot(mag_rect)

	plt.subplot(num_plots, 1, 2)
	plt.plot(mag_ham)

	plt.subplot(num_plots, 1, 3)
	plt.plot(mag_hann)

	plt.subplot(num_plots, 1, 4)
	plt.plot(mag_hanning)

	plt.subplot(num_plots, 1, 5)
	plt.plot(mag_blackman)

	plt.show()
