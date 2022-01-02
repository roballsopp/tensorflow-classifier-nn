from mido import MidiFile
import os.path
import logging
import random
import numpy as np
import importlib
from .midi_map_config import ARTICULATION_NAME_TO_IDX, NUM_ARTICULATIONS, ARTICULATION_PRECEDENCE

MuS_PER_SECOND = 1000000  # number of microseconds in a second

default_map_dir = os.path.dirname(os.path.realpath(__file__))
default_map_path = os.path.join(default_map_dir, 'default_midi_map.py')

class Markers:
	@staticmethod
	def from_file(marker_path, map_path=default_map_path):
		if not os.path.exists(map_path):
			logging.warning('No midi map found at ' + map_path + '. Using default midi map')
			map_path = default_map_path

		logging.info('Loading markers from ' + marker_path + '...')
		logging.info('Using midi map at ' + map_path)

		mod_name, file_ext = os.path.splitext(os.path.split(map_path)[-1])

		spec = importlib.util.spec_from_file_location(mod_name, map_path)
		mapper = importlib.util.module_from_spec(spec)
		spec.loader.exec_module(mapper)

		midi_map = mapper.get_map(ARTICULATION_NAME_TO_IDX)

		midi_file = MidiFile(marker_path)
		return Markers(midi_file, midi_map)

	@staticmethod
	def generate_negative_markers(positive_markers, min_distance_from_positive_markers=0):
		logging.info('Generating negative markers...')
	
		negative_markers = []

		for i, current_positive_marker in enumerate(positive_markers):
			negative_marker = None
			if len(positive_markers) > i + 1:
				next_positive_marker = positive_markers[i + 1]
				negative_marker = Markers.create_random_between(current_positive_marker, next_positive_marker, min_distance_from_positive_markers)

			if negative_marker is not None:
				negative_markers.append(negative_marker)

		logging.info(str(len(negative_markers)) + ' negative markers generated')
		return negative_markers

	@staticmethod
	def create_random_between(first_marker, second_marker, min_distance_from_markers=0):
		distance_between_markers = second_marker['pos'] - first_marker['pos'] - (min_distance_from_markers * 2)

		if distance_between_markers > 0:
			pos = (random.random() * distance_between_markers) + first_marker['pos'] + min_distance_from_markers
			return {'pos': round(pos), 'y': np.zeros(NUM_ARTICULATIONS, dtype=np.int8)}
		else:
			return None

	def __init__(self, midi_file, midi_map):
		self._ticks_per_beat = midi_file.ticks_per_beat
		self._events = midi_file.tracks[0]

		print(f'found {len(self._events)} midi events')

		if not isinstance(midi_map, tuple):
			raise TypeError('Invalid midi map. Expected list, got ' + str(type(midi_map)))

		if len(midi_map) != 128:
			raise ValueError('Invalid midi map.Expected length 128. Got length ' + str(len(midi_map)))

		self._midi_map = midi_map

	# build a map of the markers in the format:
	# {
	#  [uint samplePosition]: new Int8Array([0, 0, 1, ... NUM_ARTICULATIONS])
	#  [uint samplePosition]: new Int8Array([1, 1, 0, ... NUM_ARTICULATIONS])
	#  ...
	# }
	# where each Int8Array records which articulation(s) were hit at the samplePosition
	def get_sample_pos_map(self, sample_rate=44100):
		marker_map = {}
		currentSampPerBeat = 0
		elapsedSamples = 0

		for evt in self._events:
			numBeatsSinceLastEvt = evt.time / self._ticks_per_beat

			elapsedSamples += (numBeatsSinceLastEvt * currentSampPerBeat)

			if evt.type == 'set_tempo':
				currentSampPerBeat = (evt.tempo / MuS_PER_SECOND) * sample_rate
			elif evt.type == 'note_on' and evt.velocity != 0:
				markerPosition = round(elapsedSamples)
				mappedNote = self._midi_map[evt.note]  # map 128 possible midi notes into NUM_ARTICULATIONS
				articulations = marker_map.setdefault(markerPosition, np.zeros(NUM_ARTICULATIONS, dtype=np.int8))

				if mappedNote != ARTICULATION_NAME_TO_IDX['NO_HIT']:
					articulations[mappedNote] = 1

		return marker_map

	# build a list of the markers in the format:
	# [{
	# 	pos: uint samplePosition,
	# 	y: new Int8Array([1, 1, 0, ... NUM_ARTICULATIONS])
	# }, {
	# 	pos: uint samplePosition,
	# 	y: new Int8Array([1, 1, 0, ... NUM_ARTICULATIONS])
	# }, ...]
	# where y is an Int8Array that records which articulation(s) were hit at the samplePosition
	# OLD IMPLEMENTATION STILL USED BY ExampleBuilders
	# def get_sample_pos_list(self, sample_rate=44100):
	# 	markerMap = self.get_sample_pos_map(sample_rate)
	# 	markerArray = []
	#
	# 	for pos, y in markerMap.items():
	# 		markerArray.append({'pos': int(pos), 'y': y})
	#
	# 	markerArray.sort(key=lambda m: m['pos'])
	#
	# 	# Not exactly the number of midi events, since some happen simultaneously and are captured inside a single marker
	# 	logging.info(str(len(markerArray)) + ' markers loaded.')
	# 	return markerArray
	# SEE mido docs for a lot of the tempo stuff here: https://mido.readthedocs.io/en/latest/midi_files.html#tempo-and-beat-resolution
	def get_sample_pos_list(self, sample_rate=44100):
		marker_list = []
		# current_tempo is in microseconds per beat
		current_tempo = 0
		elapsed_mu = 0  # elapsed microsecond-ticks per beat

		for evt in self._events:
			# time is in ticks, so elapsed_mu ends up being in microsecond-ticks per beat
			#   self._ticks_per_beat is the resolution of the midi file, and remains constant through the whole thing
			#   therefore, we can defer the division by self._ticks_per_beat to later so we are working in integers only
			#   this way we do not accumulate small numerical errors by adding floats over and over
			elapsed_mu += current_tempo * evt.time

			if evt.type == 'set_tempo':
				# tempo is in microseconds per beat
				current_tempo = evt.tempo
			elif evt.type == 'note_on' and evt.velocity != 0:
				art = self._midi_map[evt.note]  # map 128 possible midi notes into NUM_ARTICULATIONS

				if art != ARTICULATION_NAME_TO_IDX['NO_HIT']:
					marker_list.append({
						# elapsed_mu / self._ticks_per_beat gets us total elapsed microseconds
						#   / MuS_PER_SECOND gets us elapsed seconds, and finally * sample_rate gets us elapsed samples
						'pos': int(round(elapsed_mu / self._ticks_per_beat / MuS_PER_SECOND * sample_rate)),
						'art': art,
						'prec': ARTICULATION_PRECEDENCE[art]
					})

		return marker_list