import numpy as np

def load(numFeatures, numLabels):
	X_metal = np.fromfile('./training_data/X_dry.dat', dtype=np.float32);
	y_metal = np.fromfile('./training_data/y_dry.dat', dtype=np.uint8);
	X_avatar = np.fromfile('./training_data/X_dry.dat', dtype=np.float32);
	y_avatar = np.fromfile('./training_data/y_dry.dat', dtype=np.uint8);

	X = np.vstack((X_metal, X_avatar));
	y = np.vstack((y_metal, y_avatar));

	X.shape = (-1, numFeatures);
	y.shape = (-1, numLabels);

	np.random.shuffle(X);
	np.random.shuffle(y);

	print(X.shape);
	print(y.shape);

	return [X[:-1000], y[:-1000], X[-1000:], y[-1000:]];