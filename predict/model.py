import tensorflow as tf
import nn


def get_data_format_string(channels_last):
	return 'NHWC' if channels_last else 'NCHW'


def fix_divide_by_zero(inputs):
	return tf.where(tf.is_nan(inputs), x=tf.zeros(inputs.shape, dtype=inputs.dtype), y=inputs)


def create_layer(inputs, size, channels_last=True, name=''):
	pad_amt = round(size / 2)
	num_bands = inputs.shape[2].value if channels_last else inputs.shape[3].value

	if channels_last:
		inputs = tf.pad(inputs, [[0, 0], [pad_amt, pad_amt], [0, 0], [0, 0]], "CONSTANT")
	else:
		inputs = tf.pad(inputs, [[0, 0], [0, 0], [pad_amt, pad_amt], [0, 0]], "CONSTANT")

	inputs = tf.nn.convolution(
		inputs,
		filter=nn.transient_kernel([size, num_bands, 1]),
		padding='VALID',
		data_format=get_data_format_string(channels_last),
		name='layer_1_' + name
	)

	return inputs[:, :-1, :, :]


def spectrogram_model(inputs, channels_last=True):
	fft_size = 32
	half_fft_size = int(fft_size / 2)
	num_output_bands = int(half_fft_size + 1)

	inputs = tf.pad(inputs, [[0, 0], [half_fft_size, 0]], "CONSTANT")

	stfts = tf.contrib.signal.stft(inputs, frame_length=fft_size, frame_step=1, fft_length=fft_size, pad_end=True)
	inputs = tf.abs(stfts)[:, :-half_fft_size, :]

	if channels_last:
		inputs = tf.reshape(inputs, [-1, num_output_bands, 1])

	# create the "batch" dim, even though there is only one example
	inputs = tf.expand_dims(inputs, axis=0)

	inputs = nn.rms_normalize_per_band(inputs, channels_last=channels_last)
	outputs = create_layer(inputs, 512, channels_last=channels_last, name='spectrogram')

	return outputs


def magnitude_model(inputs, channels_last=True):
	inputs = tf.abs(inputs)

	if channels_last:
		inputs = tf.reshape(inputs, [-1, 1, 1])

	# create the "batch" dim, even though there is only one example
	inputs = tf.expand_dims(inputs, axis=0)

	outputs = create_layer(inputs, 1024, channels_last=channels_last, name='magnitude')

	return outputs


def find_peaks(inputs, channels_last=True):
	data_format = get_data_format_string(channels_last)

	peak_filter = tf.nn.convolution(
		inputs,
		filter=nn.peak_kernel([2, 1, 1]),
		padding='SAME',
		data_format=data_format,
		name='peak_filter_1',
	)

	peaks = peak_filter / tf.abs(peak_filter)
	peaks = tf.minimum(fix_divide_by_zero(peaks), 0)

	peaks = tf.nn.convolution(
		peaks,
		filter=nn.peak_kernel([2, 1, 1]),
		padding='SAME',
		data_format=data_format,
		name='peak_filter_2',
	)

	peaks = tf.maximum(peaks * -1, 0)

	return peaks


class Model:
	def __init__(self, inputs, channels_last=True):
		spectrogram_out = spectrogram_model(inputs, channels_last=channels_last)
		magnitude_out = magnitude_model(inputs, channels_last=channels_last)

		summed_out = tf.nn.relu(nn.rms_normalize(spectrogram_out) + nn.rms_normalize(magnitude_out))

		peaks = find_peaks(summed_out, channels_last=channels_last)

		final_out = summed_out * peaks

		# remove "batch" dim, and band dim
		final_out = tf.squeeze(final_out, axis=[0, 2])
		# transpose back into channels first
		self._raw_outputs = tf.transpose(final_out)

	def get_raw(self):
		return self._raw_outputs

	def forward_prop(self):
		return tf.round(tf.nn.tanh(self._raw_outputs))
