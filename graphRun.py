import train
import predict
import loadData
import time
from argparse import ArgumentParser
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

parser = ArgumentParser()

parser.add_argument('input_pattern', help="Specify input file(s) using glob pattern")
parser.add_argument('layers',
										help="Specify nn hidden layer architecture. Provide space separated integers to specify the number of neurons in heach hidden layer.",
										nargs='*',
										type=int)

args = parser.parse_args()

logging.info('Loading files matching ' + args.input_pattern + '...')
dataset = loadData.load(args.input_pattern)
logging.info('Files loaded successfully. Loaded ' + str(dataset.num_examples) + ' training examples')

layers = [dataset.num_features] + args.layers + [dataset.num_labels]
logging.info('Training neural network with architecture ' + ', '.join(map(str, layers)) + '...')

start_time = time.time()

train.train(layers, dataset)

# X = loadData.loadWav('./test_data/435.wav')
# print('wav loaded')
#
# print('Predicting...')
#
# predictions = predict.slider([input_layer_size, 50, num_labels], X, './tmp/model_run50_23_4000.ckpt')
#

end_time = time.time()
print('Done after', end_time - start_time, 'seconds')
#
# loadData.saveWav('./test_data/mono_dry_bounced.wav', np.amax(predictions, axis=1))
#
