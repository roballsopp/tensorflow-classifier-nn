import logging
import os
import glob
from argparse import ArgumentParser

from data_gen.generator import DataGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

parser = ArgumentParser()

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
late_marker_window = args.late_marker_window
output_dir = args.output_dir

generator = DataGenerator(output_dir, num_features, num_labels, desired_num_examples, marker_offset, late_marker_window)

audio_path = os.path.join(os.getcwd(), args.audio)
marker_path = os.path.join(os.getcwd(), args.markers)

files = glob.glob(audio_path, recursive=True)

for filepath in files:
	generator.create_training_data(filepath, marker_path)
