import numpy as np
import struct
import os
import uuid
import logging

FORMAT_TO_TYPE = {
	1: np.float32,
	2: np.int32,
	3: np.int16,
	4: np.int8
}

class NdatHeader:
	def __init__(
		self,
		feature_width,
		feature_format,
		label_width,
		label_format,
		num_examples,
		feature_height=1,
		feature_channels=1,
		label_height=1,
		label_channels=1,
		label_offset=0,
	):
		self.feature_width = feature_width
		self.feature_height = feature_height
		self.feature_channels = feature_channels
		self.feature_type = FORMAT_TO_TYPE[feature_format]
		self.feature_bytes = self.feature_type(0).itemsize * feature_width
		self.label_width = label_width
		self.label_height = label_height
		self.label_channels = label_channels
		self.label_type = FORMAT_TO_TYPE[label_format]
		self.label_bytes = self.label_type(0).itemsize * label_width
		self.num_examples = num_examples
		self.example_bytes = self.feature_bytes + self.label_bytes
		self.label_offset = label_offset

	def to_bytes(self):
		return struct.pack(
			'<4sIIHHIIHHIi',
			NdatHeader.HEADER_ID.encode(),
			self.feature_width,
			self.feature_height,
			self.feature_channels,
			NdatHeader.FMT_FLOAT,
			self.label_width,
			self.label_height,
			self.label_channels,
			NdatHeader.FMT_FLOAT,
			self.num_examples,
			self.label_offset,
		)

	HEADER_SIZE = 36
	HEADER_ID = 'NDAT'
	FMT_FLOAT = 1
	FMT_INT32 = 2
	FMT_INT16 = 3
	FMT_INT8 = 4

	@staticmethod
	def from_file(file_path):
		with open(file_path, 'rb') as file:
			header_id = file.read(4)

			if header_id != b'NDAT':
				raise ValueError('Header id incorrect', header_id)

			header = struct.unpack('<IIHHIIHHIi', file.read(32))

			feature_width = header[0]
			feature_height = header[1]
			feature_channels = header[2]
			feature_format = header[3]
			label_width = header[4]
			label_height = header[5]
			label_channels = header[6]
			label_format = header[7]
			num_examples = header[8]
			label_offset = header[9]

			logging.info('--- Data header ---')
			logging.info('Feature width: ' + str(feature_width))
			logging.info('Feature height: ' + str(feature_height))
			logging.info('Feature channels: ' + str(feature_channels))
			logging.info('Feature format: ' + str(feature_format))
			logging.info('Label width: ' + str(label_width))
			logging.info('Label height: ' + str(label_height))
			logging.info('Label channels: ' + str(label_channels))
			logging.info('Label format: ' + str(label_format))
			logging.info('Num Examples: ' + str(num_examples))
			logging.info('Label offset: ' + str(label_offset))

			return NdatHeader(
				feature_width,
				feature_format,
				label_width,
				label_format,
				num_examples,
				feature_height,
				feature_channels,
				label_height,
				label_channels,
				label_offset
			)

	@staticmethod
	def from_blob(blob):
		tmp_dir = './tmp'
		if not os.path.isdir(tmp_dir):
			os.mkdir(tmp_dir)

		temp_file_path = os.path.join(tmp_dir, str(uuid.uuid1()))

		with open(temp_file_path, 'wb') as file:
			blob.download_to_file(file)

		header = NdatHeader.from_file(temp_file_path)

		os.remove(temp_file_path)

		return header
