import tensorflow as tf
import nn


def slice_at_markers(signal, markers, signal_slice_size=319):
	marker_indicies = tf.where(tf.cast(markers, dtype=tf.bool))[:, 1]

	signal_channels = signal.shape[0].value

	slices = tf.map_fn(lambda x: signal[:, x:x + signal_slice_size], marker_indicies, dtype=tf.float32)
	slices.set_shape([None, signal_channels, signal_slice_size])

	return slices


# spectrogram_size is the number of ffts to take, or the length in samples of the output signal
# fft_size will determine the number of frequency bins (fft_size / 2) + 1
def get_avg_response(signal, markers, spectrogram_size=256, fft_size=64):
	signal_slice_size = spectrogram_size + fft_size - 1

	signal_slices = slice_at_markers(signal, markers, signal_slice_size=signal_slice_size)
	stfts = tf.contrib.signal.stft(signal_slices, frame_length=fft_size, frame_step=1, fft_length=fft_size)

	magnitudes = tf.abs(stfts)
	phases = tf.angle(stfts)

	magnitudes = nn.rms_normalize(magnitudes, axis=0)

	avg_magnitude = tf.reduce_mean(magnitudes, axis=[0])
	avg_phase = tf.reduce_mean(phases, axis=[0])

	real = avg_magnitude * tf.cos(avg_phase)
	imag = avg_magnitude * tf.sin(avg_phase)

	return tf.complex(real, imag)
