SUPPORTED_ARTICULATIONS = [
	'SNARE',
	'SNARE_SIDESTICK',
	'SNARE_RIMSHOT',
	'SNARE_FLAM',
	'KICK',
	'TOM1',
	'TOM2',
	'TOM3',
	'TOM4',
	'TOM5',
	'TOM6',
	'HIHAT_CLOSED',
	'HIHAT_PEDAL',
	'HIHAT_OPEN',
	'HIHAT_OPEN_BELL',
	'CRASH1',
	'CRASH2',
	'RIDE_EDGE',
	'RIDE_BOW',
	'RIDE_BELL',
	'CHINA',
	'SPLASH',
	'COWBELL',
	'OTHER',
]

ARTICULATION_NAME_TO_IDX = {name:i for i, name in enumerate(SUPPORTED_ARTICULATIONS)}

ARTICULATION_NAME_TO_IDX["NO_HIT"] = -1
NUM_ARTICULATIONS = len(SUPPORTED_ARTICULATIONS)

ARTICULATION_NAME_TO_PRECEDENCE = {
	'SNARE': 1,
	'SNARE_SIDESTICK': 1,
	'SNARE_RIMSHOT': 1,
	'SNARE_FLAM': 1,
	'KICK': 1,
	'TOM1': 1,
	'TOM2': 1,
	'TOM3': 1,
	'TOM4': 1,
	'TOM5': 1,
	'TOM6': 1,
	'HIHAT_CLOSED': 0,
	'HIHAT_PEDAL': 0,
	'HIHAT_OPEN': 0,
	'HIHAT_OPEN_BELL': 0,
	'CRASH1': 0,
	'CRASH2': 0,
	'RIDE_EDGE': 0,
	'RIDE_BOW': 0,
	'RIDE_BELL': 0,
	'CHINA': 0,
	'SPLASH': 0,
	'COWBELL': 1,
	'OTHER': 0,
}

ARTICULATION_PRECEDENCE = [ARTICULATION_NAME_TO_PRECEDENCE[name] for name in SUPPORTED_ARTICULATIONS]
