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


def magnitude_model(inputs, channels_last=True):
	normed_inputs = pre_process(inputs, channels_last=channels_last)

	time_axis = 0 if channels_last else 1
	band_axis = 1 if channels_last else 2
	input_length = inputs.shape[time_axis].value

	# TODO: play with fft size (trade frequency reso for time reso),
	# window size (Odd window will make phase behave better (STFT class 2)),
	# and phase unwrapping (looks like phase spectrogram would be more useful for determining transients when unwrapped).
	# Different windows might work better for noise rejection (blackman/blackman-harris)
	fft_size = 255
	# TODO: handle any number for fft_step. kernel_size below must evaluate to an even number, and fft_step has an impact
	fft_step = 10

	# shift input to fft so output energy is positioned correctly for later stages
	# if we don't shift here, the fft draws energy forward in time, making our markers early when they come out later
	padding = [[0, 0], [0, 0]]
	padding[time_axis] = [fft_size - 1, 0]
	fft_inputs = tf.pad(normed_inputs, padding)

	# advantage of using fft to generate magnitude over just raw signal is the fft magnitude is separated from the phase component
	# in the raw signal, the magnitudes are all there, but the sine waves are shifted to make them very uneven
	sig_mag = tf.abs(stfts)
	stfts = spectral.stft(fft_inputs, window_length=fft_size, step=fft_step, channels_last=channels_last)

	# normalize each band. reduce along time axis to do this. very similar effect to just dropping all low freqs
	# sig_mag = nn.mean_normalize(sig_mag, axis=time_axis)
	# drop all low freqs
	# sig_mag = sig_mag[:, 15:, :]

	total_mag = tf.reduce_sum(sig_mag, axis=[band_axis])

	# the log scale really brings out the lower transients, it must also bring out the noise, but it seems pretty dang good...
	total_mag_log = tf.log(total_mag + 1e-16)

	# create the "batch" dim, even though there is only one example
	total_mag_log = tf.expand_dims(total_mag_log, axis=0)

	# TODO: this must evaluate to an even number right now
	kernel_size = int(500 / fft_step)

	mag_diff = tf.nn.convolution(
		total_mag_log,
		filter=kernels.util.expand_1d(kernels.diff(kernel_size)),
		padding='SAME',
		data_format=get_1d_data_format_string(channels_last)
	)

	# compensate for any sample skipping we did during the stft
	if fft_step > 1:
		mag_diff = audio.interpolate(mag_diff, input_length, channels_last=channels_last)

	# needed for lower fft resolutions (worked nicely at fft length of 64)
	# smoothed_gradient = nn.smooth_1d(mag_diff, size=128)

	rect_gradient = tf.nn.relu(mag_diff)

	peaks = nn.find_peaks_1d(rect_gradient, channels_last=channels_last)

	final_out = rect_gradient * peaks

	# remove "batch" dim
	final_out = tf.squeeze(final_out, axis=[0])

	return post_process(final_out, channels_last=channels_last)
