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

	# TODO: this must evaluate to an even number right now
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
	time_axis = 1 if channels_last else 2
	band_axis = 2 if channels_last else 3

	phase_sum = tf.reduce_sum(phase, axis=[band_axis])

	# TODO: this must evaluate to an even number right now
	kernel_size = int(400 / fft_step)

	phase_diff = tf.nn.convolution(
		phase_sum,
		filter=kernels.util.expand_1d(kernels.diff(kernel_size)) * -1,
		padding='SAME',
		data_format=get_1d_data_format_string(channels_last)
	)

	phase_mean, phase_var = tf.nn.moments(phase_diff, axes=[time_axis])
	phase_std = tf.sqrt(phase_var)

	# adjust so that only the peaks are positive
	phase_diff = phase_diff - (phase_std * 1)

	return phase_diff


def magnitude_model(inputs, channels_last=True):
	normed_inputs = pre_process(inputs, channels_last=channels_last)

	time_axis = 0 if channels_last else 1
	input_length = inputs.shape[time_axis].value

	# TODO: play with fft size (trade frequency reso for time reso),
	# window size (Odd window will make phase behave better (STFT class 2)),
	# and phase unwrapping (looks like phase spectrogram would be more useful for determining transients when unwrapped).
	# Different windows might work better for noise rejection (blackman/blackman-harris)
	window_size = 255
	# TODO: handle any number for fft_step. kernel_size below must evaluate to an even number, and fft_step has an impact
	fft_step = 10

	# shift input to fft so output energy is positioned correctly for later stages
	# if we don't shift here, the fft draws energy forward in time, making our markers early when they come out later
	padding = [[0, 0], [0, 0]]
	padding[time_axis] = [window_size - 1, 0]
	fft_inputs = tf.pad(normed_inputs, padding)

	# advantage of using fft to generate magnitude over just raw signal is the fft magnitude is separated from the phase component
	# in the raw signal, the magnitudes are all there, but the sine waves are shifted to make them very uneven
	stft = spectral.stft(fft_inputs, window_length=window_size, step=fft_step, channels_last=channels_last, window_fn=None)

	# add "batch" dim, even though there is only one example here
	stft = tf.expand_dims(stft, axis=0)

	sig_mag = tf.abs(stft)

	# redefine time axis now that we've added the batch dim
	time_axis = 1 if channels_last else 2
	band_axis = 2 if channels_last else 3

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

	# remove "batch" dim
	final_out = tf.squeeze(final_out, axis=[0])

	return post_process(final_out, channels_last=channels_last)
