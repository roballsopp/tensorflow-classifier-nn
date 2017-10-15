import glob
import logging
import os
from argparse import ArgumentParser

from data_gen.markers import Markers
from load import Wave
from ndat.writer import NdatWriter

from data_gen.autoencoder import ExampleBuilder as AutoencoderBuilder
from data_gen.single_label import ExampleBuilder as SingleLabelBuilder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

parser = ArgumentParser()

parser.add_argument('--single-label', action='store_true')
parser.add_argument('--audio', type=str, help='Specify the path or glob to the audio file(s)', default='**/*.wav')
parser.add_argument('--markers', type=str, help='Specify the path to the marker midi file', default='markers.mid')
parser.add_argument('--output-dir', type=str, help='Specify the output directory for the data files')
parser.add_argument('--offset', type=int, help='Specify the offset used from each marker to save the actual example data', default=0)
parser.add_argument('--num-features', type=int, help='Specify the length (samples) of each feature set', default=1024)
parser.add_argument('--num-labels', type=int, help='Specify the length of each label set', default=1024)
parser.add_argument('--late-marker-window', type=int, help='Specify how much of the end of each label set to zero out', default=0)
parser.add_argument('--num-examples', type=int, help='Specify the desired number of examples to extract from the audio', default=10000)

args = parser.parse_args()

num_features = args.num_features
num_labels = args.num_labels
desired_num_examples = args.num_examples
# how far forward in time to pad the audio signal, to effectively make the markers mark earlier than the actual transients
marker_offset = args.offset
output_dir = args.output_dir
marker_file_path = os.path.join(os.getcwd(), args.markers)

glob_audio_path = os.path.join(os.getcwd(), args.audio)
files = glob.glob(glob_audio_path, recursive=True)

def get_builder(wav_file, markers):
	if args.single_label:
		return SingleLabelBuilder(wav_file, markers, num_features, 64, marker_offset)
	else:
		return AutoencoderBuilder(wav_file, markers, desired_num_examples, num_features, 64, num_labels, marker_offset, args.late_marker_window)

for audio_file_path in files:
	audio_dir = os.path.dirname(audio_file_path)
	midi_map_file_path = os.path.join(audio_dir, 'map.py')

	wav_file = Wave.from_file(audio_file_path)

	logging.info('Audio info - Sample Rate: ' + str(wav_file.sample_rate))

	markers = Markers.from_file(marker_file_path, midi_map_file_path)

	example_builder = get_builder(wav_file, markers)
	writer = NdatWriter(example_builder)

	relative_audio_dir = os.path.relpath(audio_dir, os.getcwd())
	output_folder = os.path.join(output_dir, relative_audio_dir) if output_dir else audio_dir
	output_file_path = os.path.join(output_folder, os.path.basename(audio_file_path))

	writer.to_file(output_file_path)
