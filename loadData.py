import numpy as np

def loadX(numFeatures):
	X = np.fromfile('./training_data/X_dry.dat', dtype=np.float32)
	X.shape = (-1, numFeatures)
	return X

def loady(numLabels):
	y = np.fromfile('./training_data/y_dry.dat', dtype=np.uint8)
	y.shape = (-1, numLabels)
	return y