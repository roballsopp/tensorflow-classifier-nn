import logging
import os
from argparse import ArgumentParser

import numpy as np

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
		return (np.cos(np.linspace(np.pi, 3 * np.pi, width)) + 1) / 2, width

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


if __name__ == "__main__":
	parser = ArgumentParser()

	parser.add_argument('input_file', type=str, help='Specify the path to the marker midi file')
	parser.add_argument('--sr', type=int, help='Specify the sample rate for the output wav file', default=44100)
	parser.add_argument('--shape', type=str, help='Specify the shape of each label. Can be `point`, `diff`, or `hann`', default='point')
	parser.add_argument('--width', type=int, help='Specify the width of each label.')
	parser.add_argument('output_file', type=str, help='Specify the path to the output wav file')

	args = parser.parse_args()

	marker_file_path = os.path.join(os.getcwd(), args.input_file)
	output_file_path = os.path.join(os.getcwd(), args.output_file)
	sample_rate = args.sr
	label_shape = args.shape
	label_width = args.width

	markers = Markers.from_file(marker_file_path)
	markers_list = markers.get_sample_pos_list(sample_rate)

	wav_data = np.zeros((1, markers_list[-1]['pos'] + 100))
	output_len = wav_data.shape[1]

	label_data, width = get_shape(label_shape, label_width)
	left_lobe_width = width // 2
	right_lobe_width = width - left_lobe_width

	for marker in markers_list:
		label_center = marker['pos']

		label_start, label_end, start, end = get_shape_clip(label_center - left_lobe_width, label_center + right_lobe_width, output_len)

		wav_data[0, start:end] = label_data[label_start:label_end]

	Wave(wav_data, sample_rate).to_file(output_file_path)
