import numpy as np
import wave
import struct

def load(file_path):
	with open(file_path, 'rb') as file:
		header_id = file.read(4)

		if header_id != b'NDAT':
			raise ValueError('Header id incorrect', header_id)

		num_features, num_labels, num_examples, label_offset = struct.unpack('<IIII', file.read(16))

		example_dt = np.dtype(np.float32)
		example_dt = example_dt.newbyteorder('<')

		data_dt = np.dtype([('X', example_dt, (num_features,)), ('y', np.uint8, (num_labels,))])

		return {
			'data': np.fromfile(file, dtype=data_dt, count=num_examples),
			'num_features': num_features,
			'num_labels': num_labels,
			'num_examples': num_examples,
			'label_offset': label_offset
		}

def loadWav(filepath):
	print('Loading wav', filepath)
	file = wave.open(filepath, mode='rb')

	sample_rate = file.getframerate()
	byte_depth = file.getsampwidth()
	num_channels = file.getnchannels()
	num_samples = file.getnframes()

	print('Sample Rate:', sample_rate)
	print('Bit Depth:', byte_depth * 8)
	print('Channels:', num_channels)
	print('Num Samples:', num_samples)

	buf = file.readframes(num_samples)

	file.close()

	data = np.frombuffer(buf, dtype=np.uint8)
	data.shape = (-1, byte_depth)

	fit_to_4_bytes = np.zeros((num_samples, 4), dtype=np.uint8)
	fit_to_4_bytes[:, (4 - byte_depth):] = data

	dt = np.dtype(np.int32)
	dt = dt.newbyteorder('L')

	return np.frombuffer(fit_to_4_bytes.data, dtype=dt) / (2 ** 31)

def saveWav(filepath, data):
	byte_depth = 2
	float_to_int = 2 ** ((byte_depth * 8) - 1)
	data_int = np.rint(data * float_to_int).astype(np.int16)

	file = wave.open(filepath, 'wb')

	file.setnchannels(1)
	file.setsampwidth(byte_depth)
	file.setframerate(44100)
	file.writeframesraw(data_int.data)

	file.close()
