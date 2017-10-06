import logging
import os
from data_gen.markers import Markers
from data_gen.example_builder import ExampleBuilder
from load import Wave
from ndat.writer import NdatWriter

class DataGenerator:
	def __init__(
		self,
		output_dir,
		num_features=1024,  # in samples
		num_labels=1024,
		desired_num_examples=5000,
		marker_offset=0,
		late_marker_window=0
	):
		self._output_dir = output_dir
		self._num_features = num_features
		self._num_labels = num_labels
		self._desired_num_examples = desired_num_examples
		self._marker_offset = marker_offset
		self._late_marker_window = late_marker_window

	def create_training_data(self, audio_file_path, marker_file_path):
		audio_dir = os.path.dirname(audio_file_path)
		midi_map_file_path = os.path.join(audio_dir, 'map.py')

		wav_file = Wave.from_file(audio_file_path)
		markers = Markers.from_file(marker_file_path, midi_map_file_path)
		logging.info('Audio info - Sample Rate: ' + str(wav_file.sample_rate))

		example_builder = ExampleBuilder(wav_file, markers, self._desired_num_examples, self._num_features, self._num_labels, self._marker_offset, self._late_marker_window)
		writer = NdatWriter(example_builder)

		relative_audio_dir = os.path.relpath(audio_dir, os.getcwd())
		output_folder = os.path.join(self._output_dir, relative_audio_dir) if self._output_dir else audio_dir
		output_file_path = os.path.join(output_folder, os.path.basename(audio_file_path))

		return writer.to_file(output_file_path)
