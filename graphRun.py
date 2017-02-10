import train
import numpy as np
import predict
import loadData
import time

avatar = loadData.load('./training_data/avatar/kit1.ndat')

input_layer_size = avatar['num_features']
num_labels = avatar['num_labels']

start_time = time.time()

train.train([input_layer_size, 50, num_labels], avatar['data'], 'run50_1_4000')
# train.train([input_layer_size, 300, num_labels], data, 'run300_2000');
# train.train([input_layer_size, 500, num_labels], data, 'run500_2000');

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
