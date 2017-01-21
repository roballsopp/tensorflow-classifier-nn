import tensorflow as tf
import train
import loadData

input_layer_size  = 5513;
num_labels = 128;

data = loadData.load(input_layer_size, num_labels);

for hidden_layer_size in range(50, 500, 50):
	train.train([input_layer_size, hidden_layer_size, num_labels], data, 'run' + str(hidden_layer_size))
