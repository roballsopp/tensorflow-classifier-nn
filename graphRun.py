import train
import numpy as np
import predict
import loadData
import time

avatar1 = loadData.load('./training_data/avatar/kit1.ndat')
avatar2 = loadData.load('./training_data/avatar/kit2.ndat')
avatar3 = loadData.load('./training_data/avatar/kit3.ndat')
metal1 = loadData.load('./training_data/metal/kit1.ndat')
metal2 = loadData.load('./training_data/metal/kit2.ndat')
vintage1 = loadData.load('./training_data/vintage/kit1.ndat')
vintage2 = loadData.load('./training_data/vintage/kit2.ndat')
vintage3 = loadData.load('./training_data/vintage/kit3.ndat')

data = np.concatenate((avatar1['data'], avatar2['data'], avatar3['data'], metal1['data'], metal2['data'], vintage1['data'], vintage2['data'], vintage3['data']))

input_layer_size = avatar1['num_features']
num_labels = avatar1['num_labels']

start_time = time.time()

train.train((input_layer_size, 100, 50, num_labels), data, 'run100_50_1_4000_no_inv_offset_adam_3kits_pl')

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
