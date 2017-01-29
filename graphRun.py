# import tensorflow as tf
import train
import numpy as np
import predict
import loadData
import time

input_layer_size = 5513
num_labels = 23

# data = loadData.load(input_layer_size, num_labels);

# for hidden_layer_size in range(50, 500, 50):
# 	train.train([input_layer_size, hidden_layer_size, num_labels], data, 'run' + str(hidden_layer_size))

# train.train([input_layer_size, 50, num_labels], data, 'run50_23_4000');
# train.train([input_layer_size, 300, num_labels], data, 'run300_2000');
# train.train([input_layer_size, 500, num_labels], data, 'run500_2000');

X = loadData.loadWav('./test_data/mono_dry.wav')


print('wav loaded')

print('Predicting...')

start_time = time.time()
predictions = predict.slider([input_layer_size, 50, num_labels], X, './tmp/model_run50_23_4000.ckpt')
end_time = time.time()

# predictions = predict.slide_predict([input_layer_size, 50, num_labels], X, './tmp/model_run50_23_4000.ckpt');

print('Predicting complete after', end_time - start_time, 'seconds')

loadData.saveWav('./test_data/mono_dry_bounced.wav', np.amax(predictions, axis=1))

print('saved')