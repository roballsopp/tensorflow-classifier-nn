import numpy as np
import tensorflow as tf
import wave
import struct
import random
import os
import uuid
import logging
from google.cloud import storage

storage_client = storage.Client(project='transient-finder-training')

FORMATS = {
	1: np.float32,
	2: np.int32,
	3: np.int16,
	4: np.int8
}

class DataHeader:
	def __init__(self, num_features, feature_format, num_labels, label_format, num_examples, label_offset):
		self._num_features = num_features
		self._feature_type = FORMATS[feature_format]
		self._feature_bytes = self._feature_type(0).itemsize * num_features
		self._num_labels = num_labels
		self._label_type = FORMATS[label_format]
		self._label_bytes = self._label_type(0).itemsize * num_labels
		self._num_examples = num_examples
		self._example_bytes = self._feature_bytes + self._label_bytes
		self._label_offset = label_offset
		self._num_channels = 1

	@property
	def example_bytes(self):
		return self._example_bytes

	def parse_example(self, example_raw):
		features_raw = tf.substr(example_raw, 0, self._feature_bytes)
		labels_raw = tf.substr(example_raw, self._feature_bytes, self._label_bytes)

		features = tf.decode_raw(features_raw, self._feature_type)
		labels = tf.decode_raw(labels_raw, self._label_type)

		return tf.reshape(features, [self._num_features, self._num_channels]), tf.reshape(labels, [self._num_labels])

	@staticmethod
	def from_file(file_path):
		with open(file_path, 'rb') as file:
			header_id = file.read(4)

			if header_id != b'NDAT':
				raise ValueError('Header id incorrect', header_id)

			num_features, feature_format, num_labels, label_format, num_examples, label_offset = struct.unpack('<IHIHIi', file.read(20))

			return DataHeader(num_features, feature_format, num_labels, label_format, num_examples, label_offset)

	@staticmethod
	def from_blob(blob):
		tmp_dir = './tmp'
		if not os.path.isdir(tmp_dir):
			os.mkdir(tmp_dir)

		temp_file_path = os.path.join(tmp_dir, str(uuid.uuid1()))

		with open(temp_file_path, 'wb') as file:
			blob.download_to_file(file)

		# TODO: cleanup the temp file to save money on gcs
		return DataHeader.from_file(temp_file_path)


def from_filenames(filenames):
	random.shuffle(filenames)
	num_files = len(filenames)

	if num_files == 0:
		raise ValueError('No files found')

	header = DataHeader.from_file(filenames[0])

	dataset = tf.contrib.data.FixedLengthRecordDataset(filenames, header.example_bytes, header_bytes=24)
	dataset = dataset.map(header.parse_example, num_threads=8, output_buffer_size=50000)
	return dataset


def from_bucket(bucket_name, prefix=None):
	data_bucket = storage.bucket.Bucket(storage_client, bucket_name)

	filenames = []
	blobs = []

	for blob in data_bucket.list_blobs(prefix=prefix):
		if blob.name.endswith('/'):
			continue

		blobs.append(blob)
		path = 'gs://' + data_bucket.name + '/' + blob.name
		filenames.append(path)

	num_files = len(filenames)

	if num_files == 0:
		raise ValueError('No files found')

	logging.info('Found ' + str(num_files) + ' files.')

	header = DataHeader.from_blob(blobs[0])

	dataset = tf.contrib.data.FixedLengthRecordDataset(filenames, header.example_bytes, header_bytes=24)
	dataset = dataset.map(header.parse_example, num_threads=8, output_buffer_size=50000)
	return dataset

def load_wav(filepath):
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

def save_wav(filepath, data):
	byte_depth = 2
	float_to_int = 2 ** ((byte_depth * 8) - 1)
	data_int = np.rint(data * float_to_int).astype(np.int16)

	file = wave.open(filepath, 'wb')

	file.setnchannels(1)
	file.setsampwidth(byte_depth)
	file.setframerate(44100)
	file.writeframesraw(data_int.data)

	file.close()
