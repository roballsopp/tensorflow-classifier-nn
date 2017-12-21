import tensorflow as tf

def stft(inputs, fft_length=64, step=1, pad_end=False, channels_last=False):
	# fft requires sample data in the last dimension, so transpose if its not
	if channels_last:
		inputs = tf.transpose(inputs)

	stfts = tf.contrib.signal.stft(inputs, frame_length=fft_length, frame_step=step, fft_length=fft_length, pad_end=pad_end)

	if channels_last:
		stfts = tf.transpose(stfts, perm=[1, 2, 0])

	return stfts
