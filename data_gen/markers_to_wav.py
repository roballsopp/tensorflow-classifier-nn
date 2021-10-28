import logging
import os
from argparse import ArgumentParser

import numpy as np

from data_gen.markers import Markers
from load import Wave

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

if __name__ == "__main__":
	parser = ArgumentParser()

	parser.add_argument('input_file', type=str, help='Specify the path to the marker midi file')
	parser.add_argument('--sr', type=int, help='Specify the sample rate for the output wav file', default=44100)
	parser.add_argument('output_file', type=str, help='Specify the path to the output wav file')

	args = parser.parse_args()

	marker_file_path = os.path.join(os.getcwd(), args.input_file)
	output_file_path = os.path.join(os.getcwd(), args.output_file)
	sample_rate = args.sr

	markers = Markers.from_file(marker_file_path)
	markers_list = markers.get_sample_pos_list(sample_rate)

	wav_data = np.zeros((1, markers_list[-1]['pos'] + 100))

	for marker in markers_list:
		wav_data[0, marker['pos']] = 1.

	Wave(wav_data, sample_rate).to_file(output_file_path)
