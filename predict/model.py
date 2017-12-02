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

	smoothing_size = 128

	inputs = tf.pad(inputs, [[0, 0], [smoothing_size, 0], [0, 0], [0, 0]], 'CONSTANT')

	inputs = tf.layers.average_pooling2d(
		inputs,
		pool_size=[smoothing_size, 1],
		strides=[1, 1],
		padding='same'
	)[:, smoothing_size:, :, :]

	return inputs


def spectrogram_model(inputs, channels_last=True):
	fft_size = 64
	shift_amt = fft_size - 1
	num_output_bands = int((fft_size / 2) + 1)

	inputs = tf.pad(inputs, [[0, 0], [shift_amt, 0]], 'CONSTANT')

	stfts = tf.contrib.signal.stft(inputs, frame_length=fft_size, frame_step=1, fft_length=fft_size, pad_end=True)
	inputs = tf.abs(stfts)

	if channels_last:
		inputs = tf.reshape(inputs, [-1, num_output_bands, 1])

	# create the "batch" dim, even though there is only one example
	inputs = tf.expand_dims(inputs, axis=0)

	# inputs = nn.rms_normalize_per_band(inputs, channels_last=channels_last)
	outputs = create_layer(inputs, 1024, channels_last=channels_last, name='spectrogram')

	return outputs[:, shift_amt:, :, :]


def widen_labels(inputs, channels_last=True):
	if channels_last:
		inputs = tf.reshape(inputs, [-1, 1, 1])
	else:
		inputs = tf.reshape(inputs, [1, -1, 1])

	# create the "batch" dim, even though there is only one example
	inputs = tf.expand_dims(inputs, axis=0)

	outputs = tf.nn.convolution(
		inputs,
		filter=nn.blur_kernel([50, 1, 1, 1]),
		padding='SAME',
		data_format=get_data_format_string(channels_last),
		name='widening_filter',
	)

	# remove "batch" dim, and band dim
	outputs = tf.squeeze(outputs, axis=[0, 2])
	# transpose back into channels first
	outputs = tf.transpose(outputs)

	return nn.normalize(outputs)

class Model:
	def __init__(self, inputs, channels_last=True):
		spectrogram_out = spectrogram_model(inputs, channels_last=channels_last)

		summed_out = tf.nn.relu(nn.rms_normalize(spectrogram_out))

		peaks = nn.find_peaks(summed_out, channels_last=channels_last)

		final_out = summed_out * peaks

		# remove "batch" dim, and band dim
		final_out = tf.squeeze(final_out, axis=[0, 2])
		# transpose back into channels first
		self._raw_outputs = tf.transpose(final_out)

	def get_raw(self):
		return self._raw_outputs

	def forward_prop(self):
		return tf.round(tf.nn.tanh(self._raw_outputs))

	@staticmethod
	def cost(predictions, labels):
		predictions = widen_labels(predictions)
		labels = widen_labels(labels)

		return tf.sqrt(tf.reduce_mean(tf.square(labels - predictions)))

