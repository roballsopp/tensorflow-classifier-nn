import tensorflow as tf
import nn
import nn.kernels as kernels
import nn.kernels.util
import spectral


def get_data_format_string(channels_last):
	return 'NHWC' if channels_last else 'NCHW'


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

	# TODO: play with fft size (trade frequency reso for time reso),
	# window size (Odd window will make phase behave better (STFT class 2)),
	# and phase unwrapping (looks like phase spectrogram would be more useful for determining transients when unwrapped).
	# Different windows might work better for noise rejection (blackman/blackman-harris)
	fft_size = 256

	# shift input to fft so output energy is positioned correctly for later stages
	# if we don't shift here, the fft draws energy forward in time, making our markers early when they come out later
	padding = [[0, 0], [0, 0]]
	padding[time_axis] = [fft_size - 1, 0]
	fft_inputs = tf.pad(normed_inputs, padding)

	# advantage of using fft to generate magnitude over just raw signal is the fft magnitude is separated from the phase component
	# in the raw signal, the magnitudes are all there, but the sine waves are shifted to make them very uneven
	stfts = nn.stft(fft_inputs, fft_length=fft_size, step=1, channels_last=channels_last)
	sig_mag = tf.abs(stfts)

	# normalize each band. reduce along time axis to do this. very similar effect to just dropping all low freqs
	# sig_mag = nn.mean_normalize(sig_mag, axis=time_axis)
	# drop all low freqs
	# sig_mag = sig_mag[:, 15:, :]

	total_mag = tf.reduce_sum(sig_mag, axis=[band_axis])

	# the log scale really brings out the lower transients, it must also bring out the noise, but it seems pretty dang good...
	total_mag_log = tf.log(total_mag + 1e-16)

	# create the "batch" dim, even though there is only one example
	total_mag_log = tf.expand_dims(total_mag_log, axis=0)

	mag_diff = tf.nn.convolution(
		total_mag_log,
		filter=kernels.util.expand_1d(kernels.diff(512)),
		padding='SAME',
		data_format=get_1d_data_format_string(channels_last)
	)

	# needed for lower fft resolutions (worked nicely at fft length of 64)
	# smoothed_gradient = nn.smooth_1d(mag_diff, size=128)

	rect_gradient = tf.nn.relu(mag_diff)

	peaks = nn.find_peaks_1d(rect_gradient, channels_last=channels_last)

	final_out = rect_gradient * peaks

	# remove "batch" dim
	final_out = tf.squeeze(final_out, axis=[0])

	return post_process(final_out, channels_last=channels_last)


def autocorrelation_model(inputs, avg_response, channels_last=True):
	signal_filter = tf.abs(avg_response)
	filter_length = signal_filter.shape[0].value
	fft_length = (filter_length - 1) * 2

	# create a "height" dim
	signal_filter = tf.expand_dims(signal_filter, axis=[0])

	stfts = nn.stft(inputs, fft_length=fft_length, step=1, pad_end=True, channels_last=channels_last)
	sig_mag = tf.abs(stfts)

	# create the "batch" dim, even though there is only one example
	sig_mag = tf.expand_dims(sig_mag, axis=0)

	correlated_out = tf.nn.convolution(
		sig_mag,
		filter=tf.expand_dims(signal_filter, axis=3),
		padding='VALID',
		data_format=get_data_format_string(channels_last)
	)

	smoothed_out = nn.smooth(correlated_out, size=128)

	normed_out = nn.rms_normalize(smoothed_out)

	peaks = nn.find_peaks_1d(normed_out, channels_last=channels_last)

	final_out = normed_out * peaks

	# remove "batch" dim, and band dim
	final_out = tf.squeeze(final_out, axis=[0, 2])

	return final_out


def autocorrelation2_model(inputs, signal_filter, channels_last=True):
	# create the "batch" dim, even though there is only one example
	inputs = tf.expand_dims(inputs, axis=0)

	inputs = tf.pad(inputs, [[0, 0], [0, 255], [0, 0]], 'CONSTANT')

	correlated_out = tf.nn.convolution(
		inputs,
		filter=tf.expand_dims(signal_filter, axis=2),
		padding='VALID',
		data_format=get_1d_data_format_string(channels_last)
	)

	smoothed_out = nn.smooth(correlated_out, size=128)

	normed_out = nn.rms_normalize(smoothed_out)

	peaks = nn.find_peaks_1d(normed_out, channels_last=channels_last)

	final_out = normed_out * peaks

	# remove "batch" dim
	final_out = tf.squeeze(final_out, axis=[0])

	return final_out


def deconvolution_model(inputs, deconv_filter, channels_last=True):
	filter_length = deconv_filter.shape[0].value
	fft_length = (filter_length - 1) * 2

	stfts = nn.stft(inputs, fft_length=fft_length, step=1, channels_last=channels_last)
	sig_mag = tf.abs(stfts)
	sig_phase = tf.angle(stfts)

	# sig_mag = tf.Print(sig_mag, [sig_mag, sig_phase], summarize=50)

	deconv_filter = tf.expand_dims(deconv_filter, axis=[0])

	filter_mag = tf.abs(deconv_filter)
	filter_phase = tf.angle(deconv_filter)

	# filter_mag = tf.Print(filter_mag, [filter_mag, filter_phase], summarize=50)

	mag_out = sig_mag / filter_mag
	phase_out = sig_phase - filter_phase

	# mag_out = tf.Print(mag_out, [mag_out, phase_out], summarize=50)

	stfts = spectral.polar_to_complex([mag_out, phase_out])

	outputs = spectral.istft(stfts, fft_length=fft_length, channels_last=channels_last)

	return outputs
