import tensorflow as tf
import nn
import nn.kernels as kernels
import nn.kernels.util
import spectral
import audio


def get_1d_data_format_string(channels_last):
	return 'NWC' if channels_last else 'NCW'


def widen_labels(inputs, channels_last=True):
	# create the "batch" dim, even though there is only one example
	inputs = tf.expand_dims(inputs, axis=0)

	outputs = tf.nn.convolution(
		inputs,
		filter=kernels.util.expand_1d(kernels.blur(50)),
		padding='SAME',
		data_format=get_1d_data_format_string(channels_last),
		name='widening_filter',
	)

	# remove "batch" dim
	outputs = tf.squeeze(outputs, axis=[0])

	return nn.normalize(outputs)


def calc_cost(predictions, labels, channels_last=True):
	predictions = widen_labels(predictions, channels_last=channels_last)
	labels = widen_labels(labels, channels_last=channels_last)

	return tf.reduce_mean(tf.abs(labels - predictions))


def batch_inputs(x, axis, frame_length, overlap):
	return tf.contrib.signal.frame(
		x,
		frame_length=frame_length,
		frame_step=frame_length - overlap,
		pad_end=True,
		axis=axis
	)


# requires input of shape [..., batches, batch_length]
def unbatch_outputs(batch, frame_length, overlap):
	return tf.contrib.signal.overlap_and_add(
		batch,
		frame_step=frame_length - overlap
	)


def pre_process(inputs, channels_last=True):
	time_axis = 0 if channels_last else 1
	# remove dc offset
	dc_offset = tf.reduce_mean(inputs, axis=[time_axis], keep_dims=True)
	inputs = inputs - dc_offset

	# make sure signals come in roughly at the same energy level (rms of 1)
	# rms_norm_factor = tf.sqrt((inputs.shape[time_axis].value * (new_rms ** 2)) / tf.reduce_sum(inputs ** 2, axis=time_axis))
	normed_inputs = nn.rms_normalize(inputs, axis=time_axis)

	return normed_inputs


def post_process(outputs, channels_last=True):
	time_axis = 0 if channels_last else 1
	# everything exits with an rms of 1
	return nn.rms_normalize(outputs, axis=time_axis, non_zero=True)


def shift_tensor(x, shift_amt, axis):
	ndims = x.shape.ndims

	start_from = [slice(None)] * ndims
	start_from[axis] = slice(shift_amt, None)

	end_padding = [[0, 0]] * ndims
	end_padding[axis] = [0, shift_amt]

	x = tf.pad(x[start_from], end_padding)

	return x

# assumes mag is a 4d tensor (batch, height, width, chan) or (b, c, h, w)
def find_magnitude_peaks(mag, fft_step, channels_last=True):
	band_axis = 2 if channels_last else 3

	mag_sum = tf.reduce_sum(mag, axis=[band_axis])

	kernel_size = int(500 / fft_step)

	mag_diff = tf.nn.convolution(
		mag_sum,
		filter=kernels.util.expand_1d(kernels.diff(kernel_size)),
		padding='SAME',
		data_format=get_1d_data_format_string(channels_last)
	)

	smooth_size = int(round(60 / fft_step))
	mag_diff = nn.smooth_1d(mag_diff, size=smooth_size)

	return mag_diff


# assumes phase is a 4d tensor (batch, height, width, chan) or (b, c, h, w)
def find_phase_peaks(phase, fft_step, channels_last=True):
	band_axis = 2 if channels_last else 3

	phase_sum = tf.reduce_sum(phase, axis=[band_axis])

	smooth_size = int(round(300 / fft_step))
	phase_sum = nn.smooth_1d(phase_sum, size=smooth_size)

	kernel_size = int(round(500 / fft_step))

	phase_diff = tf.nn.convolution(
		phase_sum,
		filter=kernels.util.expand_1d(kernels.diff(kernel_size)) * -1,
		padding='SAME',
		data_format=get_1d_data_format_string(channels_last)
	)

	# TODO: verify this finds std across batch dim
	phase_mean, phase_var = tf.nn.moments(phase_diff, axes=list(range(phase_diff.shape.ndims)))
	phase_std = tf.sqrt(phase_var)

	# adjust so that only the peaks are positive
	phase_diff = phase_diff - (phase_std * 1)

	return phase_diff


def batched_stft(x, window_size, step, channels_last=True):
	time_axis = 0 if channels_last else 1
	input_length = x.shape[time_axis].value

	# shift input to fft so output energy is positioned correctly for later stages
	# if we don't shift here, the fft draws energy forward in time, making our markers early when they come out later
	padding = [[0, 0], [0, 0]]
	padding[time_axis] = [window_size - 1, 0]
	normed_inputs = tf.pad(x, padding)
	num_ffts_per_batch = 5000
	# (num_ffts_per_batch * fft_step) + (window_size - 1) ensures there is exactly enough space to get num_ffts_per_batch and not skip any samples
	batch_length = (num_ffts_per_batch * step) + (window_size - 1)
	batch_overlap = window_size - 1
	batched_inputs = batch_inputs(normed_inputs, axis=time_axis, frame_length=batch_length, overlap=batch_overlap)

	# stft requires time axis to be last always
	if channels_last:
		batched_inputs = tf.transpose(batched_inputs, perm=[0, 2, 1])

	# advantage of using fft to generate magnitude over just raw signal is the fft magnitude is separated from the phase component
	# in the raw signal, the magnitudes are all there, but the sine waves are shifted to make them very uneven
	stft = tf.contrib.signal.stft(batched_inputs, frame_length=window_size, frame_step=step, window_fn=None)

	if channels_last:
		stft = tf.transpose(stft, perm=[0, 2, 3, 1])

	# redefine time axis since we added a batch dim now
	time_axis = 1 if channels_last else 2

	stft_shape = stft.shape.as_list()
	stft_shape[time_axis] = stft_shape[time_axis] * stft_shape[0]
	# leave batch dim intact since later ops are designed for batch processing
	stft_shape[0] = 1

	stft = tf.reshape(stft, stft_shape)

	# TODO: what if it evaluates to a decimal?
	step_adjusted_length = int(input_length / step)
	up_to_batch_padding = [slice(None), slice(None), slice(None), slice(None)]
	up_to_batch_padding[time_axis] = slice(step_adjusted_length)
	# get rid of padding added by batching op
	stft = stft[up_to_batch_padding]

	return stft


def magnitude_model(inputs, channels_last=True):
	normed_inputs = pre_process(inputs, channels_last=channels_last)

	window_size = 511
	fft_step = 10
	time_axis = 0 if channels_last else 1
	input_length = normed_inputs.shape[time_axis].value

	stft = batched_stft(normed_inputs, window_size, fft_step, channels_last=channels_last)

	# redefine time axis since we added a batch dim now
	time_axis = 1 if channels_last else 2
	band_axis = 2 if channels_last else 3

	sig_mag = tf.abs(stft)

	sig_phase = tf.angle(stft)
	sig_phase = spectral.unwrap(sig_phase, axis=band_axis)

	# phase needs to be shifted backward in time by half of the fft length relative
	# to the magnitude spectrum in order to line up after diffing
	phase_shift_amt = int(round((window_size / fft_step) / 2))

	sig_phase = shift_tensor(sig_phase, phase_shift_amt, axis=time_axis)

	phase_diff = find_phase_peaks(sig_phase, fft_step, channels_last)
	mag_diff = find_magnitude_peaks(sig_mag, fft_step, channels_last)

	mag_phase = nn.abs_mean_normalize(mag_diff) + nn.abs_mean_normalize(phase_diff)

	# normalize each band. reduce along time axis to do this. very similar effect to just dropping all low freqs
	# sig_mag = nn.mean_normalize(sig_mag, axis=time_axis)
	# drop all low freqs
	# sig_mag = sig_mag[:, 15:, :]

	# compensate for any sample skipping we did during the stft
	if fft_step > 1:
		mag_phase = audio.interpolate(mag_phase, input_length, channels_last=channels_last)

	rect_gradient = tf.nn.relu(mag_phase)

	peaks = nn.find_peaks_1d(rect_gradient, channels_last=channels_last)

	final_out = rect_gradient * peaks

	# remove batch dim
	final_out = tf.squeeze(final_out, axis=[0])

	return post_process(final_out, channels_last=channels_last)
