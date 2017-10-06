import logging
import math
import numpy as np

def convert_markers_to_label_array(markers, length):
	label_buffer = np.zeros((length,), np.float32)

	for marker in markers:
		label_buffer[marker['pos']] = 1

	return label_buffer

# calculate a position offset that evenly spaces examples through the available
# length of the audio, overlapping or spacing examples as necessary
def calc_example_offset(feature_width, num_examples, available_space):
	needed_space = feature_width * num_examples
	overlap = (needed_space - available_space) / (num_examples - 1)
	offset = feature_width - overlap

	if offset < 1:
		logging.warning('Not enough space available in audio for ' + str(num_examples) + ' unique examples.')
		return 1

	return math.floor(offset)

def calc_num_examples_from_offset(feature_width, example_offset, available_space):
	return math.floor((available_space - feature_width + example_offset) / example_offset)

class ExampleBuilder:
	# late_marker_window is how many samples from the end of an example we will remove markers
	# if a marker occurs very close to the end of an example, there may not be enough data
	# for the neural net to actually learn anything from
	def __init__(self, wav_file, markers, desired_num_examples, feature_width, label_width, marker_offset=0, late_marker_window=0):
		label_width = label_width or feature_width
		# delay the audio by some amount so the markers are early
		self._audio_buffer = np.pad(wav_file.get_chan(0), (marker_offset, 0), 'constant')

		# we might train the autoencoder to output a different number of labels than features for performance reasons
		label_feature_ratio = label_width / feature_width
		# when we are calculating the positions of the markers, well need to use this ratio'd sample rate
		label_sample_rate = label_feature_ratio * wav_file.sample_rate
		label_buffer_length = math.ceil(label_feature_ratio * len(self._audio_buffer))
		markers_list = markers.get_sample_pos_list(label_sample_rate)

		# dropWithinSamples = labelSampleRate * 0.02 # drop markers within 20 ms of another marker
		# filteredMarkersList = Markers.filterByDominant(markers_list, dropWithinSamples)

		# logging.info('Removed ${markers_list.length - filteredMarkersList.length} markers that were within 20 ms of a preceeding marker.')

		self._label_buffer = convert_markers_to_label_array(markers_list, label_buffer_length)
		# number of samples to use for each training example
		self._feature_width = feature_width
		self._label_width = label_width
		self._feature_set_offset = calc_example_offset(feature_width, desired_num_examples, len(self._audio_buffer))
		self._label_set_offset = self._feature_set_offset * label_feature_ratio
		self._num_examples = calc_num_examples_from_offset(feature_width, self._feature_set_offset, len(self._audio_buffer))
		self._label_offset = marker_offset
		self._late_marker_window = late_marker_window
		self._current_example = 0

		# this might print a negative number, in which case there is actually space between the examples rather than overlap
		logging.info('Example overlap is ' + str(feature_width - self._feature_set_offset))

	@property
	def num_examples(self):
		return self._num_examples

	@property
	def feature_width(self):
		return self._feature_width

	@property
	def label_width(self):
		return self._label_width

	@property
	def label_offset(self):
		return self._label_offset

	def has_next(self):
		return self._current_example < self._num_examples

	def get_next_example(self):
		feature_set_start_pos = int(self._current_example * self._feature_set_offset)
		feature_set_end_pos = int(feature_set_start_pos + self._feature_width)
		features = self._audio_buffer[feature_set_start_pos:feature_set_end_pos]

		label_set_start_pos = int(self._current_example * self._label_set_offset)
		label_set_end_pos = int(label_set_start_pos + self._label_width)
		labels = self._label_buffer[label_set_start_pos:label_set_end_pos]

		# if late_marker_window is not 0, get rid of that many samples of labels at the end of the example
		if self._late_marker_window != 0:
			labels[-self._late_marker_window:] = 0

		self._current_example += 1

		return features, labels

	def reset(self):
		self._current_example = 0
