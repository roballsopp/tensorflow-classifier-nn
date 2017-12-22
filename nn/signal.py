import tensorflow as tf

def stft(inputs, window_length=64, step=1, pad_end=False, channels_last=False):
	# fft requires sample data in the last dimension, so transpose if its not
	if channels_last:
		inputs = tf.transpose(inputs)

	window_fn = functools.partial(tf.contrib.signal.hamming_window, periodic=True)

	stfts = tf.contrib.signal.stft(inputs, frame_length=window_length, frame_step=step, pad_end=pad_end, window_fn=window_fn)

	if channels_last:
		stfts = tf.transpose(stfts, perm=[1, 2, 0])

	return stfts
