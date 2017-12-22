import tensorflow as tf


def stft(inputs, fft_length=64, step=1, pad_end=False, channels_last=False):
	# fft requires sample data in the last dimension, so transpose if its not
	if channels_last:
		inputs = tf.transpose(inputs)

	stfts = tf.contrib.signal.stft(inputs, frame_length=fft_length, frame_step=step, fft_length=fft_length, pad_end=pad_end)

	if channels_last:
		stfts = tf.transpose(stfts, perm=[1, 2, 0])

	return stfts


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
