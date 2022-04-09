import logging
import os
from argparse import ArgumentParser

import numpy as np
import scipy.signal

from data_gen.markers import Markers
from load import Wave

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def get_shape(name, width):
	if name == 'point':
		return np.ones((1,)), 1

	if name == 'diff':
		if type(width) is not int:
			raise ValueError('--width argument is required with --shape="diff"')
		return np.sin(np.linspace(-np.pi, np.pi, width)), width

	if name == 'hann':
		if type(width) is not int:
			raise ValueError('--width argument is required with --shape="hann"')
		return np.hanning(width), width

	if name == 'half-hann':
		if type(width) is not int:
			raise ValueError('--width argument is required with --shape="half-hann"')
		shape = np.hanning(width)
		half_width = width // 2
		shape[:half_width] = 0 # first half is zeros
		return shape, width

	if name == 'gauss':
		if type(width) is not int:
			raise ValueError('--width argument is required with --shape="gauss"')
		return scipy.signal.windows.gaussian(width, width/6), width

	raise ValueError(f'unrecognized label shape: {name}')

def get_shape_clip(start, end, max_len):
	clip_start = 0
	clip_end = end - start
	if start < 0:
		clip_start = -start
		start = 0

	if end >= max_len:
		clip_end -= end + 1 - max_len
		end = max_len - 1

	return clip_start, clip_end, start, end

# `behind_window` and `ahead_window`: how much territory in samples, either before or after, a primary hit controls.
#   If a non-primary (like a cymbal) is in a primary's territory, it will be scaled down based on how close it is.
#   The search window is asymmetric because a cymbal hit in the relative quiet just before a snare hit is much more obvious
#   than a cymbal hit that occurs during the decay of a snare hit
# Note: the two hits in a flam are about 2000 samps apart at 44100
class PrimaryWindowWeighting:
	def __init__(self, markers, behind_window=3000, ahead_window=1000):
		self.behind_window = behind_window
		self.ahead_window = ahead_window
		self.markers = markers

	def get_weight(self, marker_idx):
		base_marker = self.markers[marker_idx]

		# if this hit has a primary precedence, its weight it always 1
		if base_marker['prec'] == 1:
			return 1.

		p_behind, _ = self._find_primary_behind(marker_idx)
		p_ahead, _ = self._find_primary_ahead(marker_idx)

		if p_behind is None and p_ahead is None:
			return 1.

		if p_behind is not None and p_ahead is None:
			dist = base_marker['pos'] - p_behind['pos']
			# the closer it is, the lower the weight. index 0 in the hanning window is 0, and index primary_window would be 1
			return np.hanning(self.behind_window * 2)[dist]

		if p_behind is None and p_ahead is not None:
			dist = p_ahead['pos'] - base_marker['pos']
			return np.hanning(self.ahead_window * 2)[dist]

		behind_dist = base_marker['pos'] - p_behind['pos']
		ahead_dist = p_ahead['pos'] - base_marker['pos']

		if behind_dist < ahead_dist:
			return np.hanning(self.behind_window * 2)[behind_dist]

		return np.hanning(self.ahead_window * 2)[ahead_dist]

	def _find_primary_behind(self, start):
		base_marker = self.markers[start]
		for i in range(start - 1, -1, -1):
			m = self.markers[i]
			# we're already outside the window, no primary found
			if base_marker['pos'] - m['pos'] > self.behind_window:
				return None, i
			# cool, we found a primary within the window!
			if m['prec'] == 1:
				return m, i

		return None, start

	def _find_primary_ahead(self, start):
		base_marker = self.markers[start]
		for i in range(start + 1, len(markers_list)):
			m = self.markers[i]
			# we're already outside the window, no primary found
			if m['pos'] - base_marker['pos'] > self.ahead_window:
				return None, i
			# cool, we found a primary within the window!
			if m['prec'] == 1:
				return m, i

		return None, start

# just pick the earliest onset in the window
class EarliestOnsetWeighting:
	def __init__(self, markers, window=3000):
		self.window = window
		self.markers = markers
		self.weights = np.hanning(window * 2)[window:]
		self.last_marker_pos = 0
		self.last_marker_weight = 0

	def get_weight(self, marker_idx):
		marker = self.markers[marker_idx]
		distance_from_last = marker['pos'] - self.last_marker_pos

		self.last_marker_pos = marker['pos']

		if distance_from_last >= self.window:
			self.last_marker_weight = 1.
		else:
			self.last_marker_weight = 1. - self.weights[distance_from_last]

		return self.last_marker_weight


if __name__ == "__main__":
	parser = ArgumentParser()

	parser.add_argument('input_file', type=str, help='Specify the path to the marker midi file')
	parser.add_argument('--sr', type=int, help='Specify the sample rate for the output wav file', default=44100)
	parser.add_argument('--shape', type=str, help='Specify the shape of each label. Can be `point`, `diff`, `hann`, or `half-hann`', default='point')
	parser.add_argument('--width', type=int, help='Specify the width of each label.')
	parser.add_argument('output_file', type=str, help='Specify the path to the output wav file')

	args = parser.parse_args()

	marker_file_path = os.path.join(os.getcwd(), args.input_file)
	output_file_path = os.path.join(os.getcwd(), args.output_file)
	sample_rate = args.sr
	label_shape = args.shape
	label_width = args.width

	default_map_dir = os.path.dirname(os.path.realpath(__file__))
	# change to default_midi_map.py for drums, default_piano_midi_map.py for piano
	default_map_path = os.path.join(default_map_dir, 'default_midi_map.py')
	# default_map_path = os.path.join(default_map_dir, 'default_piano_midi_map.py')

	markers = Markers.from_file(marker_file_path, map_path=default_map_path)
	markers_list = markers.get_sample_pos_list(sample_rate)

	# re: label_width below. technically we'd only need to pad the end by half the label_width rounding up, but it
	#   doesn't matter much as long as the padding is enough for the last full label.
	# BEWARE that the label file will not be the same length as the original audio signal! We don't supply the original
	#   signal so there is currently no way to crop/pad the label signal correctly
	# NOTE that the beginning of the label file does align with the beginning of the original audio signal. No need
	#   to worry about offsetting the label signal correctly, only need to pad or crop the end
	wav_data = np.zeros((1, markers_list[-1]['pos'] + label_width))
	output_len = wav_data.shape[1]

	label_shape_data, width = get_shape(label_shape, label_width)
	left_lobe_width = width // 2
	right_lobe_width = width - left_lobe_width

	# EarliestOnsetWeighting is good for piano, PrimaryWindowWeighting good for drums
	# weighting_scheme = EarliestOnsetWeighting(markers_list, window=4000)
	weighting_scheme = PrimaryWindowWeighting(markers_list, behind_window=3000, ahead_window=1000)

	for i, marker in enumerate(markers_list):
		label_center = marker['pos']

		label_start, label_end, start, end = get_shape_clip(label_center - left_lobe_width, label_center + right_lobe_width, output_len)

		weight = weighting_scheme.get_weight(i)

		scaled_label = label_shape_data[label_start:label_end] * weight

		wav_data[0, start:end] = np.maximum(scaled_label, wav_data[0, start:end])

	Wave(wav_data, sample_rate).to_file(output_file_path)
