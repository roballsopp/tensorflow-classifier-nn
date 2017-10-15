import logging
import numpy as np
from data_gen.spectrogram import spectrogram
from data_gen.markers import Markers

def normalize(signal):
	range = np.ptp(signal)
	mean = np.mean(signal)
	return (signal - mean) / range

class ExampleBuilder:
	def __init__(self, wav_file, markers, feature_width, fft_size, marker_offset=0, with_timeseries=True):
		self._fft_size = fft_size

		signal = wav_file.get_chan(0)

		# push signal back by length of fft to make sure the true start of events is in view when slices corresponding to markers are taken
		padded_signal = np.pad(signal, [[fft_size, 0]], mode='constant')

		logging.info('Creating spectrogram...')
		spec = spectrogram(padded_signal, size=fft_size, sample_rate=wav_file.sample_rate)

		spec = normalize(spec)

		logging.info('Spectrogram complete!')

		if with_timeseries:
			trimmed_len = len(padded_signal) + 1 - fft_size
			trimmed_signal = padded_signal[:trimmed_len]
			trimmed_signal = normalize(trimmed_signal)
			trimmed_signal.shape = (-1, 1)

			self._feature_buffer = np.concatenate([trimmed_signal, spec], axis=1)
		else:
			self._feature_buffer = spec

		positive_markers = markers.get_sample_pos_list(wav_file.sample_rate)
		negative_markers = Markers.generate_negative_markers(positive_markers, min_distance_from_positive_markers=25)
		self._markers_list = positive_markers + negative_markers
		self._markers_list.sort(key=lambda m: m['pos'])

		# number of samples to use for each training example
		self._feature_width = feature_width
		self._feature_height = self._feature_buffer.shape[1]
		self._label_width = int(1)
		self._num_examples = len(self._markers_list)
		self._label_offset = marker_offset
		self._current_example = 0

	@property
	def num_examples(self):
		return self._num_examples

	@property
	def feature_width(self):
		return self._feature_width

	@property
	def feature_height(self):
		return self._feature_height

	@property
	def label_width(self):
		return self._label_width

	@property
	def label_offset(self):
		return self._label_offset

	def has_next(self):
		return self._current_example < self._num_examples

	def get_next_example(self):
		current_marker = self._markers_list[self._current_example]
		feature_set_start_pos = current_marker['pos']
		feature_set_end_pos = current_marker['pos'] + self._feature_width
		features = self._feature_buffer[feature_set_start_pos:feature_set_end_pos]

		one_label = np.array([1], dtype=np.float32)
		zero_label = np.array([0], dtype=np.float32)

		label = one_label if np.sum(current_marker['y']) > 0 else zero_label

		self._current_example += 1

		return features, label

	def reset(self):
		self._current_example = 0
