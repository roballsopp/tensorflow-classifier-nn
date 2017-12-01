import tensorflow as tf

def rms_normalize_per_band(spectrogram, channels_last=True):
	band_axis = 1 if channels_last else 2
	band_rms = tf.sqrt(tf.reduce_mean(tf.square(spectrogram), axis=[band_axis]))
	den = tf.expand_dims(band_rms, axis=1)

	return spectrogram / den


def rms_normalize(inputs):
	rms = tf.sqrt(tf.reduce_mean(tf.square(inputs)))
	return inputs / rms


def normalize(inputs):
	maximum = tf.reduce_max(tf.abs(inputs))
	return inputs / maximum