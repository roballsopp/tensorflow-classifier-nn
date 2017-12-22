import numpy as np
import tensorflow as tf
import wave
import logging
import struct

BYTE_DEPTH_TO_TYPE = {
	1: np.int8,
	2: np.int16
}

class Wave:
	def __init__(self, data, sample_rate=44100):
		self._data = data.astype(np.float32)
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
		float_to_int = (2 ** (bit_depth - 1)) - 1
		data_int = np.rint(self._data * float_to_int).astype(np.int16)

		data_cont = np.ascontiguousarray(data_int.T)

		with wave.open(filepath, mode='wb') as file:
			file.setnchannels(len(self._data))
			file.setsampwidth(int(bit_depth / 8))
			file.setframerate(self._sample_rate)
			file.writeframesraw(data_cont.data)

	@staticmethod
	def from_file(filepath):
		logging.info('Loading wav ' + filepath)

		with wave.open(filepath, mode='rb') as file:
			sample_rate = file.getframerate()
			byte_depth = file.getsampwidth()
			bit_depth = byte_depth * 8
			num_channels = file.getnchannels()
			num_samples = file.getnframes()

			logging.info('Sample Rate: ' + str(sample_rate))
			logging.info('Bit Depth: ' + str(bit_depth))
			logging.info('Channels: ' + str(num_channels))
			logging.info('Num Samples: ' + str(num_samples))

			buf = file.readframes(num_samples)

		dt = np.dtype(BYTE_DEPTH_TO_TYPE[byte_depth])
		dt = dt.newbyteorder('L')

		data = np.frombuffer(buf, dtype=dt)
		# buffer comes out in channels-last order
		data.shape = (num_samples, num_channels)

		# data.T to make it return data in channels-first order
		return Wave(data.T / (2 ** (bit_depth - 1)), sample_rate)


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
		float_to_int = (2 ** (bit_depth - 1)) - 1
		data_int = np.rint(self._data * float_to_int).astype(np.int16)

		data_cont = np.ascontiguousarray(data_int.T)

		with wave.open(filepath, mode='wb') as file:
			file.setnchannels(len(self._data))
			file.setsampwidth(int(bit_depth / 8))
			file.setframerate(self._sample_rate)
			file.writeframesraw(data_cont.data)

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
