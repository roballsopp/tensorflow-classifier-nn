import numpy as np
import wave
import struct
import glob

class DataSet:
	def __init__(self, data, label_offset=0):
		self.data = data
		self.label_offset = label_offset

	@property
	def features(self):
		return self.data['X']

	@property
	def labels(self):
		return self.data['y']

	@property
	def num_features(self):
		return self.data['X'].shape[1]

	@property
	def num_labels(self):
		return self.data['y'].shape[1]

	@property
	def num_examples(self):
		return self.data.shape[0]

	def shuffle(self):
		np.random.shuffle(self.data)
		return self

	@staticmethod
	def from_file(file_path):
		with open(file_path, 'rb') as file:
			header_id = file.read(4)

			if header_id != b'NDAT':
				raise ValueError('Header id incorrect', header_id)

			num_features, num_labels, num_examples, label_offset = struct.unpack('<IIIi', file.read(16))

			example_dt = np.dtype(np.float32)
			example_dt = example_dt.newbyteorder('<')

			data_dt = np.dtype([('X', example_dt, (num_features,)), ('y', np.uint8, (num_labels,))])

			data = np.fromfile(file, dtype=data_dt, count=num_examples)

			return DataSet(data, label_offset)

	@staticmethod
	def merge(datasets):
		if len(datasets) < 2:
			raise ValueError('You must provide at least two datasets to merge')

		validation_dataset = datasets[0]
		data_arrays = []

		for dataset in datasets:
			if dataset.label_offset != validation_dataset.label_offset:
				raise ValueError('Datasets must have the same label offset to be merged')

			data_arrays.append(dataset.data)

		return DataSet(np.concatenate(data_arrays), validation_dataset.label_offset)

def load(glob_pattern):
	filenames = glob.glob(glob_pattern, recursive=True)

	if len(filenames) == 1:
		return DataSet.from_file(filenames[0])

	if len(filenames) == 0:
		raise ValueError('No files found')

	datasets = []

	for filename in filenames:
		datasets.append(DataSet.from_file(filename))

	return DataSet.merge(datasets)


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
