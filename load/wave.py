import numpy as np
import tensorflow as tf
import logging
import struct

from scipy.io import wavfile as sp_wavfile

def to_float(audio):
	if audio.dtype == np.float32 or audio.dtype == np.float64:
		return audio

	if audio.dtype == np.int16:
		return audio / 2**15

	if audio.dtype == np.int32:
		return audio / 2**31

	raise ValueError(f'unsupported audio bit depth: {audio.dtype}')

class Wave:
	def __init__(self, data, sample_rate=44100):
		self._data = data
		self._sample_rate = sample_rate

	def get_data(self):
		return self._data

	def get_chan(self, chan):
		return self._data[chan]

	@property
	def num_chan(self):
		return len(self._data)

	@property
	def sample_rate(self):
		return self._sample_rate

	def to_file(self, filepath, bit_depth=16):
		if bit_depth != 16:
			raise ValueError('can only export audio at bit depth of 16 currently')
		float_to_int = (2 ** (bit_depth - 1)) - 1
		data_int = np.rint(self._data * float_to_int).astype(np.int16)

		# scipy needs channels last format
		sp_wavfile.write(filepath, self._sample_rate, data_int.T)

	@staticmethod
	def from_file(filepath):
		logging.info('Loading wav ' + filepath)
		# scipy outputs channels last
		sr, audio = sp_wavfile.read(filepath)
		audio = to_float(audio)
		return Wave(audio.T, sr)


BIT_DEPTH_TYPE = {
	8: tf.int8,
	16: tf.int16,
	# TODO: might not always be a float, check the audio_fmt
	32: tf.float32
}


class WaveTF:
	def __init__(self, data, sample_rate=44100):
		self._data = tf.convert_to_tensor(data, tf.float32)
		self._sample_rate = sample_rate

	def get_data(self):
		return self._data

	def get_chan(self, chan):
		return self._data[chan]

	@property
	def num_chan(self):
		return self._data.shape[0].value

	@property
	def sample_rate(self):
		return self._sample_rate

	# TODO: rewrite this to use tf.write_file
	def to_file(self, filepath, bit_depth=16):
		if bit_depth != 16:
			raise ValueError('can only export audio at bit depth of 16 currently')
		float_to_int = (2 ** (bit_depth - 1)) - 1
		data_int = np.rint(self._data * float_to_int).astype(np.int16)

		sp_wavfile.write(filepath, self._sample_rate, data_int)

	@staticmethod
	def from_file(filepath):
		logging.info('Loading wav ' + filepath)

		def read_chunk_header(file):
			chunk_id = file.read(4)
			chunk_size, = struct.unpack('<I', file.read(4))
			return chunk_id, chunk_size

		with open(filepath, 'rb') as file:
			riff_id, riff_size = read_chunk_header(file)

			if riff_id != b'RIFF':
				raise ValueError('RIFF header id incorrect', riff_id)

			# check if you want, but pretty much ignore
			wave_id = file.read(4)

			fmt_id, fmt_size = read_chunk_header(file)

			if fmt_id != b'fmt ':
				raise ValueError('FMT header id incorrect', riff_id)

			fmt_chunk = file.read(16)
			audio_fmt, num_chan, sr, bytes_per_sec, block_align, bit_depth = struct.unpack('<HHIIHH', fmt_chunk)

			# skip over the extended format stuff for now
			fmt_ext = file.read(fmt_size - 16)

			next_chunk_id, next_chunk_size = read_chunk_header(file)

			while next_chunk_id != b'data':
				file.seek(next_chunk_size, 1)
				next_chunk_id, next_chunk_size = read_chunk_header(file)

			data_start = file.tell()
			data_len = next_chunk_size
			byte_depth = bit_depth / 8
			num_frames = int(data_len / byte_depth / num_chan)

			file = tf.read_file(filepath)

			audio_data = tf.decode_raw(tf.substr(file, data_start, data_len), BIT_DEPTH_TYPE[bit_depth], little_endian=True)
			audio_data = tf.cast(audio_data, tf.float32) / (2 ** (bit_depth - 1))
			audio_data = tf.reshape(audio_data, [num_frames, num_chan])

			return WaveTF(tf.transpose(audio_data), sr)
