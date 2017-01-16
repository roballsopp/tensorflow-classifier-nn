from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os

tf.logging.set_verbosity(tf.logging.INFO)

input_layer_size  = 5513;
hidden_layer_size = 500;
num_labels = 128;

X = np.fromfile('X_dry.dat', dtype=np.float32)
y = np.fromfile('y_dry.dat', dtype=np.uint8)

X.shape = (-1, input_layer_size)
y.shape = (-1, num_labels)

y = np.argmax(y, 1)

X_val = X[-500:]
y_val = y[-500:]

X = X[:-500]
y = y[:-500]

# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column('', dimension=input_layer_size)]

cwd = os.getcwd()

validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    X_val,
    y_val,
    every_n_steps=50)

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[hidden_layer_size],
                                            n_classes=num_labels,
                                            model_dir=cwd + '/tmp/model',
                                            config=tf.contrib.learn.RunConfig(
                                                save_checkpoints_secs=60))

# Fit model.
classifier.fit(x=X, y=y, steps=500, monitors=[validation_monitor])

# Evaluate accuracy.
accuracy_score = classifier.evaluate(x=X_val,
                                     y=y_val)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))
