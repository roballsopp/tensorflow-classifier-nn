import tensorflow as tf
import functools
import numpy as np

default_window_fn = functools.partial(tf.contrib.signal.hamming_window, periodic=True)

def stft(inputs, window_length=64, step=1, pad_end=False, channels_last=False, window_fn=default_window_fn):
	# fft requires sample data in the last dimension, so transpose if its not
	if channels_last:
		inputs = tf.transpose(inputs)

	stft = tf.contrib.signal.stft(inputs, frame_length=window_length, frame_step=step, pad_end=pad_end, window_fn=window_fn)

	if channels_last:
		stft = tf.transpose(stft, perm=[1, 2, 0])

	return stft


def istft(stfts, fft_length=64, step=1, channels_last=False):
	# fft requires sample data in the last dimension, so transpose if its not
	if channels_last:
		perm = list(range(stfts.get_shape().ndims))
		perm.insert(0, perm.pop())
		stfts = tf.transpose(stfts, perm=perm)

	signal = tf.contrib.signal.inverse_stft(
		stfts,
		frame_length=fft_length,
		frame_step=step,
		fft_length=fft_length
	)

	if channels_last:
		signal = tf.transpose(signal)

	return signal


def complex_to_polar(complex):
	magnitude = tf.abs(complex)
	phase = tf.angle(complex)
	return magnitude, phase


def polar_to_rect(polar):
	magnitude, phase = polar

	real = magnitude * tf.cos(phase)
	imag = magnitude * tf.sin(phase)

	return real, imag


def polar_to_complex(polar):
	real, imag = polar_to_rect(polar)
	return tf.complex(real, imag)


def unwrap(p, axis=0):
	ndims = p.shape.ndims
	from_2nd = [slice(None)] * ndims
	from_2nd[axis] = slice(1, None)

	to_2nd_to_last = [slice(None)] * ndims
	to_2nd_to_last[axis] = slice(None, -1)

	diff = p[from_2nd] - p[to_2nd_to_last]

	diff_mod = (diff + np.pi) % (2 * np.pi) - np.pi
	cond = (diff_mod == -np.pi) & (diff > 0)
	diff_mod = tf.where(cond, tf.constant(np.pi, shape=diff.shape), diff_mod)
	ph_correct = diff_mod - diff
	ph_correct = tf.where(tf.abs(diff) < np.pi, tf.zeros(diff.shape), ph_correct)

	cumsum = tf.cumsum(ph_correct, axis=axis)

	to_2nd = [slice(None)] * ndims
	to_2nd[axis] = slice(None, 1)

	return tf.concat([p[to_2nd], p[from_2nd] + cumsum], axis=axis)
