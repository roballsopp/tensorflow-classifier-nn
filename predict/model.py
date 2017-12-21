import tensorflow as tf
import nn
import nn.kernels as kernels
import nn.kernels.util


def get_data_format_string(channels_last):
	return 'NHWC' if channels_last else 'NCHW'


def widen_labels(inputs, channels_last=True):
	# create the "batch" dim, even though there is only one example
	inputs = tf.expand_dims(inputs, axis=0)
	# create the "width" dim, even though there is only height
	inputs = tf.expand_dims(inputs, axis=2)

	outputs = tf.nn.convolution(
		inputs,
		filter=kernels.util.expand_2d(kernels.blur(50), width=1),
		padding='SAME',
		data_format=get_data_format_string(channels_last),
		name='widening_filter',
	)

	# remove "batch" dim, and band dim
	outputs = tf.squeeze(outputs, axis=[0, 2])

	return nn.normalize(outputs)


def calc_cost(predictions, labels, channels_last=True):
	predictions = widen_labels(predictions, channels_last=channels_last)
	labels = widen_labels(labels, channels_last=channels_last)

	return tf.sqrt(tf.reduce_mean(tf.square(labels - predictions)))


def magnitude_model(inputs, channels_last=True):
	# TODO: play with fft size (trade frequency reso for time reso),
	# window size (Odd window will make phase behave better (STFT class 2)),
	# and phase unwrapping (looks like phase spectrogram would be mroe useful for determinning transients when unwrapped).
	# Different windows might work better for noise rejection (blackman/blackman-harris
	stfts = nn.stft(inputs, fft_length=64, step=1, pad_end=True, channels_last=channels_last)
	sig_mag = tf.abs(stfts)

	# create the "batch" dim, even though there is only one example
	sig_mag = tf.expand_dims(sig_mag, axis=0)

	# inputs = nn.rms_normalize_per_band(inputs, channels_last=channels_last)

	# the log scale really brings out the lower transients, it must also bring out the noise, but it seems pretty dang good...
	mag_band_gradients = tf.log(sig_mag + 1e-16)

	band_axis = 2 if channels_last else 3

	# TODO summing after the difference convolution is the same as summing before in this case, this is inefficient
	total_mag_gradients = tf.expand_dims(tf.reduce_sum(mag_band_gradients, axis=[band_axis]), axis=band_axis)

	smoothed_gradient = tf.nn.convolution(
		total_mag_gradients,
		filter=kernels.util.expand_2d(kernels.diff(512), width=1),
		padding='SAME',
		data_format=get_data_format_string(channels_last),
		name='layer_2'
	)

	smoothed_gradient = nn.smooth(smoothed_gradient, size=128)

	normed_out = tf.nn.relu(nn.rms_normalize(smoothed_gradient))

	peaks = nn.find_peaks(normed_out, channels_last=channels_last)

	final_out = normed_out * peaks

	# remove "batch" dim, and band dim
	final_out = tf.squeeze(final_out, axis=[0, 2])

	return final_out


def autocorrelation_model(inputs, avg_response, channels_last=True):
	signal_filter = tf.abs(avg_response)

	stfts = nn.stft(inputs, fft_length=64, step=1, pad_end=True, channels_last=channels_last)
	sig_mag = tf.abs(stfts)

	# create the "batch" dim, even though there is only one example
	sig_mag = tf.expand_dims(sig_mag, axis=0)

	sig_mag = tf.pad(sig_mag, [[0, 0], [0, 127], [0, 0], [0, 0]], 'CONSTANT')

	correlated_out = tf.nn.convolution(
		sig_mag,
		filter=tf.expand_dims(signal_filter, axis=3),
		padding='VALID',
		data_format=get_data_format_string(channels_last)
	)

	smoothed_out = nn.smooth(correlated_out, size=128)

	normed_out = nn.rms_normalize(smoothed_out)

	peaks = nn.find_peaks(normed_out, channels_last=channels_last)

	final_out = normed_out * peaks

	# remove "batch" dim, and band dim
	final_out = tf.squeeze(final_out, axis=[0, 2])

	return final_out
