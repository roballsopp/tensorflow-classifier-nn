import tensorflow as tf
import nn


def get_data_format_string(channels_last):
	return 'NHWC' if channels_last else 'NCHW'


def create_layer(inputs, size, channels_last=True, name=''):
	band_axis = 2 if channels_last else 3

	inputs = tf.pow(inputs, 0.6)

	inputs = tf.nn.convolution(
		inputs,
		filter=nn.transient_kernel([2, 1, 1, 1]),
		padding='SAME',
		data_format=get_data_format_string(channels_last),
		name='layer_1_' + name
	)

	inputs = tf.expand_dims(tf.reduce_sum(tf.abs(inputs), axis=[band_axis]), axis=band_axis)

	inputs = tf.nn.convolution(
		inputs,
		filter=nn.transient_kernel([size, 1, 1, 1]),
		padding='SAME',
		data_format=get_data_format_string(channels_last),
		name='layer_2_' + name
	)


	return inputs


def spectrogram_model(inputs, channels_last=True):
	stfts = nn.stft(inputs, fft_length=64, step=1, pad_end=True, channels_last=channels_last)
	inputs = tf.abs(stfts)

	# create the "batch" dim, even though there is only one example
	inputs = tf.expand_dims(inputs, axis=0)

	# inputs = nn.rms_normalize_per_band(inputs, channels_last=channels_last)
	outputs = create_layer(inputs, 1024, channels_last=channels_last, name='spectrogram')

	return outputs


def widen_labels(inputs, channels_last=True):
	if channels_last:
		inputs = tf.reshape(inputs, [-1, 1, 1])
	else:
		inputs = tf.reshape(inputs, [1, -1, 1])
def autocorrelation_model(inputs, signal_filter, channels_last=True):
	stfts = nn.stft(inputs, fft_length=64, step=1, pad_end=True, channels_last=channels_last)
	inputs = tf.abs(stfts)

	# create the "batch" dim, even though there is only one example
	inputs = tf.expand_dims(inputs, axis=0)

	inputs = tf.pad(inputs, [[0, 0], [0, 127], [0, 0], [0, 0]], 'CONSTANT')

	outputs = tf.nn.convolution(
		inputs,
		filter=tf.expand_dims(signal_filter, axis=3),
		padding='VALID',
		data_format=get_data_format_string(channels_last)
	)

	outputs = nn.smooth(outputs, size=128)

	return outputs
def widen_labels(inputs, channels_last=True):
	# create the "batch" dim, even though there is only one example
	inputs = tf.expand_dims(inputs, axis=0)
	# create the "width" dim, even though there is only height
	inputs = tf.expand_dims(inputs, axis=2)

	outputs = tf.nn.convolution(
		inputs,
		filter=nn.blur_kernel([50, 1, 1, 1]),
		padding='SAME',
		data_format=get_data_format_string(channels_last),
		name='widening_filter',
	)

	# remove "batch" dim, and band dim
	outputs = tf.squeeze(outputs, axis=[0, 2])

	return nn.normalize(outputs)

class Model:
		spectrogram_out = spectrogram_model(inputs, channels_last=channels_last)
	def __init__(self, inputs, avg_response, channels_last=True):
		avg_magnitude = tf.abs(avg_response)
		spectrogram_out = autocorrelation_model(inputs, avg_magnitude, channels_last=channels_last)

		summed_out = tf.nn.relu(nn.rms_normalize(spectrogram_out))

		peaks = nn.find_peaks(summed_out, channels_last=channels_last)

		final_out = summed_out * peaks

		# remove "batch" dim, and band dim
		final_out = tf.squeeze(final_out, axis=[0, 2])

		self._raw_outputs = final_out

	def get_raw(self):
		return self._raw_outputs

	def forward_prop(self):
		return tf.round(tf.nn.tanh(self._raw_outputs))

	@staticmethod
	def cost(predictions, labels, channels_last=True):
		predictions = widen_labels(predictions, channels_last=channels_last)
		labels = widen_labels(labels, channels_last=channels_last)

		return tf.sqrt(tf.reduce_mean(tf.square(labels - predictions)))

