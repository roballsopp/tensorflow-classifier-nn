import train
import predict
import loadData
import time

dataset = loadData.load('./training_data/**/*.ndat')

input_layer_size = dataset.num_features
num_labels = dataset.num_labels

start_time = time.time()

train.train((input_layer_size, 100, 50, num_labels), dataset, 'run100_50_1_4000_no_inv_offset_adam_glob')

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
