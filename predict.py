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


def predict(layers, x, model_path, overlap=1000):
	num_features = layers[0]
	num_labels = layers[-1]

	graph = tf.Graph()
	with graph.as_default():

		tf_x = tf.placeholder(tf.float32, name='X')
		net = nn.FullyConnected(layers)
		inputs = build_inputs(tf_x, num_features, overlap)
		outputs = net.forward_prop(inputs)
		
		# TODO: upsample outputs and reshape back into 1 dimensional array

		sess = tf.Session(graph=graph)
		saver = net.get_saver()
		saver.restore(sess, model_path)

		predictions = sess.run(outputs, feed_dict={tf_x: x})
		sess.close()

		return predictions

args = parser.parse_args()

layers = args.layers
logging.info('Predicting with architecture ' + ', '.join(map(str, layers)) + '...')

logging.info('Loading wav data...')
X = loadData.loadWav(args.input)
logging.info('Wav loaded')

logging.info('Predicting...')
predictions = predict(layers, X, './tmp/1764_100_50_1 - 2017-02-6_23-13-15/model', args.skip)
logging.info('Prediction complete.')

logging.info('Saving predictions...')
loadData.saveWav('./test_data/435_bounced.wav', np.amax(predictions, axis=1))
logging.info('Predictions saved!')

# take a plain array (x) and convert it into slices of size num_features, possibly overlapping them
def build_inputs(x, num_features, overlap):
	x_length = x.shape.as_list()[0]
	num_features = tf.constant(num_features, name="num_features")
	overlap = tf.constant(overlap, name="overlap")
	base_size = num_features - overlap
	remainder = x_length % base_size

	def pad_input():
		input_padding = tf.zeros([base_size - remainder], name="input_padding")
		return tf.concat([x, input_padding], 0, name="append_input_padding")

	x = tf.cond(remainder > 0, pad_input, lambda: x)
	x = tf.reshape(x, [-1, base_size])

	overlap_rows = x[1:, 0:overlap]
	overlap_padding = tf.zeros([1, overlap], name="overlap_padding")
	overlap_rows = tf.concat([overlap_rows, overlap_padding], 0, name="append_overlap_padding")

	return tf.concat([x, overlap_rows], 1, name="create_final_inputs")
