import tensorflow as tf
import nn


def slice_at_markers(signal, markers, signal_slice_size):
	marker_indicies = tf.where(tf.cast(markers, dtype=tf.bool))[:, 1]

	signal_channels = signal.shape[0].value

	# marker_indicies = tf.convert_to_tensor([marker_indicies[0], marker_indicies[3], marker_indicies[9], marker_indicies[10]])

	slices = tf.map_fn(lambda x: signal[:, x:x + signal_slice_size], marker_indicies, dtype=tf.float32)
	slices.set_shape([None, signal_channels, signal_slice_size])

	return slices


# spectrogram_size is the number of ffts to take, or the length in samples of the output signal
# fft_size will determine the number of frequency bins (fft_size / 2) + 1
def get_avg_spectrogram(signal, markers, spectrogram_size=256, fft_size=64):
	signal_slice_size = spectrogram_size + fft_size - 1

	signal_slices = slice_at_markers(signal, markers, signal_slice_size=signal_slice_size)
	# signal_slices = nn.rms_normalize(signal_slices, axis=0)
	avg_signal = tf.reduce_mean(signal_slices, axis=0)

	stfts = tf.contrib.signal.stft(avg_signal, frame_length=fft_size, frame_step=1, fft_length=fft_size)

	magnitudes = tf.abs(stfts)

	return magnitudes


def get_avg_signal(signal, markers, output_length=256):
	signal_slices = slice_at_markers(signal, markers, signal_slice_size=output_length)
	signal_slices = nn.rms_normalize(signal_slices, axis=-1)
	avg_signal = tf.reduce_mean(signal_slices, axis=[0])

	return avg_signal


# spectrogram_size is the number of ffts to take, or the length in samples of the output signal
# fft_size will determine the number of frequency bins (fft_size / 2) + 1
def get_avg_response(signal, markers, fft_size=256):
	signal_slices = slice_at_markers(signal, markers, signal_slice_size=fft_size)
	signal_slices = nn.rms_normalize(signal_slices, axis=-1)
	avg_signal = tf.reduce_mean(signal_slices, axis=[0])

	stfts = tf.contrib.signal.stft(avg_signal, frame_length=fft_size, frame_step=fft_size, fft_length=fft_size)

	# we only took one dft per slice, so get rid of the 'frames' dim
	stfts = tf.squeeze(stfts, axis=[1])

	return stfts
