import numpy as np
import tensorflow as tf
import wave
import struct
import glob

FORMATS = {
	1: np.float32,
	2: np.int32,
	3: np.int16,
	4: np.int8
}

def load(glob_pattern):
	filenames = glob.glob(glob_pattern, recursive=True)

	if len(filenames) == 1:
		return from_file(filenames[0])

	if len(filenames) == 0:
		raise ValueError('No files found')

	dataset = None

	for filename in filenames:
		if dataset is None:
			dataset = from_file(filename)
		else:
			dataset = dataset.concatenate(from_file(filename))

	return dataset


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


def from_file(file_path):
	with open(file_path, 'rb') as file:
		header_id = file.read(4)

		if header_id != b'NDAT':
			raise ValueError('Header id incorrect', header_id)

		num_features, feature_format, num_labels, label_format, num_examples, label_offset = struct.unpack('<IHIHIi', file.read(20))

		feature_type = FORMATS[feature_format]
		label_type = FORMATS[label_format]

		feature_bytes = feature_type(0).itemsize * num_features
		label_bytes = label_type(0).itemsize * num_labels

		total_example_bytes = feature_bytes + label_bytes

		def parse_example(example_raw):
			features_raw = tf.substr(example_raw, 0, feature_bytes)
			labels_raw = tf.substr(example_raw, feature_bytes, label_bytes)

			features = tf.decode_raw(features_raw, feature_type)
			features.set_shape([num_features])

			labels = tf.decode_raw(labels_raw, label_type)
			labels.set_shape([num_labels])

			return features, labels

		dataset = tf.contrib.data.FixedLengthRecordDataset([file_path], total_example_bytes, header_bytes=24)
		dataset = dataset.map(parse_example)
		return dataset