import logging
import random

import tensorflow as tf
from google.cloud import storage

from ndat.header import NdatHeader

def get_example_parser(header, channels_last=False):
	def parse_example(example_raw):
		features_raw = tf.substr(example_raw, 0, header.feature_bytes)
		labels_raw = tf.substr(example_raw, header.feature_bytes, header.label_bytes)

		features = tf.decode_raw(features_raw, header.feature_type)
		labels = tf.decode_raw(labels_raw, header.label_type)

		if channels_last:
			features = tf.reshape(features, [header.feature_height, header.feature_width, header.feature_channels])
			labels = tf.reshape(labels, [header.label_width])

			timeseries_features = tf.squeeze(features[:1, :, :], axis=[0])
			spectrogram_features = features[1:, :, :]
		else:
			features = tf.reshape(features, [header.feature_channels, header.feature_height, header.feature_width])
			labels = tf.reshape(labels, [header.label_width])

			timeseries_features = tf.squeeze(features[:, :1, :], axis=[1])
			spectrogram_features = features[:, 1:, :]

		# normalize each example
		# timeseries_max = tf.reduce_max(tf.abs(timeseries_features))
		timeseries_rms = tf.sqrt(tf.reduce_mean(tf.square(timeseries_features)))
		timeseries_features = timeseries_features / timeseries_rms

		# spectrogram_max = tf.reduce_max(tf.abs(spectrogram_features))
		spectrogram_rms = tf.sqrt(tf.reduce_mean(tf.square(spectrogram_features)))
		spectrogram_features = spectrogram_features / spectrogram_rms

		return timeseries_features, spectrogram_features, labels

	return parse_example

def get_file_interleaver(header):
	def interleave_files(filename):
		return tf.data.FixedLengthRecordDataset(filename, header.example_bytes, header_bytes=NdatHeader.HEADER_SIZE)
	return interleave_files


def from_filenames(filenames, channels_last=False):
	num_files = len(filenames)

	if num_files == 0:
		raise ValueError('No files found')

	header = NdatHeader.from_file(filenames[0])

	interleaver = get_file_interleaver(header)

	filenames_dataset = tf.data.Dataset.from_tensor_slices(filenames)

	dataset = filenames_dataset.interleave(interleaver, cycle_length=num_files, block_length=1)
	dataset = dataset.map(get_example_parser(header, channels_last), num_parallel_calls=8).prefetch(30000)
	return dataset


storage_client = storage.Client(project='transient-finder-training')


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

	header = NdatHeader.from_blob(blobs[0])

	dataset = tf.data.FixedLengthRecordDataset(filenames, header.example_bytes, header_bytes=NdatHeader.HEADER_SIZE)
	dataset = dataset.map(get_example_parser(header), num_parallel_calls=8).prefetch(30000)
	return dataset