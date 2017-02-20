import tensorflow as tf
import numpy as np
import logging
from argparse import ArgumentParser
import nn
import loadData

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

parser = ArgumentParser()

parser.add_argument('input', help="Specify input file path")
parser.add_argument('layers',
										help="Specify nn hidden layer architecture. Provide space separated integers to specify the number of neurons in heach hidden layer.",
										nargs='*',
										type=int)
parser.add_argument('-s', '--skip', type=int, help="Specify how many samples between each window.", default=50)


def slider(layers, x, model_path, skip=50):
	input_layer_size = layers[0]
	num_labels = layers[-1]
	slice_size = ((x.size // input_layer_size) * input_layer_size) - input_layer_size

	graph = tf.Graph()
	with graph.as_default():

		tf_x = tf.placeholder(tf.float32, name='X')
		net = nn.FullyConnected(layers)

		output_acc = []

		for i in range(0, input_layer_size, skip):
			tf_x_shifted = tf.slice(tf_x, [i], [slice_size])
			tf_x_matrix = tf.reshape(tf_x_shifted, [-1, input_layer_size])
			prediction = net.forward_prop(tf_x_matrix)
			output_acc.append(prediction)

		output = tf.reshape(tf.concat(output_acc, 1), [-1, num_labels])

		sess = tf.Session(graph=graph)
		saver = net.get_saver()
		saver.restore(sess, model_path)

		predictions = sess.run(output, feed_dict={tf_x: x})
		sess.close()

		return predictions

args = parser.parse_args()

layers = args.layers
logging.info('Predicting with architecture ' + ', '.join(map(str, layers)) + '...')

logging.info('Loading wav data...')
X = loadData.loadWav(args.input)
logging.info('Wav loaded')

logging.info('Predicting...')
predictions = slider(layers, X, './tmp/1764_100_50_1 - 2017-02-6_23-13-15/model', args.skip)
logging.info('Prediction complete.')

logging.info('Saving predictions...')
loadData.saveWav('./test_data/435_bounced.wav', np.amax(predictions, axis=1))
logging.info('Predictions saved!')
