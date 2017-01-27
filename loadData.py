import numpy as np
import wave

def load(numFeatures, numLabels):
	X_metal = np.fromfile('./training_data/X_dry.dat', dtype=np.float32);
	y_metal = np.fromfile('./training_data/y_dry.dat', dtype=np.uint8);
	X_avatar = np.fromfile('./training_data/avatar/X_dry.dat', dtype=np.float32);
	y_avatar = np.fromfile('./training_data/avatar/y_dry.dat', dtype=np.uint8);

	X_metal.shape = (-1, numFeatures);
	y_metal.shape = (-1, numLabels);

	X_avatar.shape = (-1, numFeatures);
	y_avatar.shape = (-1, numLabels);

	X = np.vstack((X_metal, X_avatar));
	y = np.vstack((y_metal, y_avatar));

	data = np.hstack((X, y));

	np.random.shuffle(data);

	X, y = np.hsplit(data, np.array([numFeatures]));

	print('X', X.shape);
	print('y', y.shape);

	return [X[:-1000], y[:-1000], X[-1000:], y[-1000:]];

def loadWav(filepath):
	print('Loading wav', filepath);
	file = wave.open(filepath, mode='rb');

	sample_rate = file.getframerate();
	byte_depth = file.getsampwidth();
	num_channels = file.getnchannels();
	num_samples = file.getnframes();

	print('Sample Rate:', sample_rate);
	print('Bit Depth:', byte_depth * 8);
	print('Channels:', num_channels);
	print('Num Samples:', num_samples);

	buf = file.readframes(num_samples);

	file.close();

	data = np.frombuffer(buf, dtype=np.uint8);
	data.shape = (-1, byte_depth);

	fit_to_4_bytes = np.zeros((num_samples, 4), dtype=np.uint8);
	fit_to_4_bytes[:, (4 - byte_depth):] = data;

	dt = np.dtype(np.int32);
	dt = dt.newbyteorder('L');

	return np.frombuffer(fit_to_4_bytes.data, dtype=dt) / (2 ** 31);

def saveWav(filepath, data):
	byte_depth = 2;
	float_to_int = 2 ** ((byte_depth * 8) - 1);
	data_int = np.rint(data * float_to_int).astype(np.int16);

	file = wave.open(filepath, 'wb');

	file.setnchannels(1);
	file.setsampwidth(byte_depth);
	file.setframerate(44100);
	file.writeframesraw(data_int.data);

	file.close();