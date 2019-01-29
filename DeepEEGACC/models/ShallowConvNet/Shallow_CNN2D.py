import sys
sys.path.insert(0, "/home/cirl/Amir/Human-Activity-EEG-Accelerometer")
import numpy as np
import os
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
import time

import tensorflow as tf
import random as rn
from keras import backend as K, optimizers
from DeepEEGACC.input_preparation import build_inputs
from keras.callbacks import EarlyStopping, CSVLogger
from DeepEEG.evaluation import compute_accuracy, evalRes
from keras.utils.vis_utils import plot_model
from keras.constraints import max_norm
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import AveragePooling2D
from keras.layers import Input
os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
# https://pdfs.semanticscholar.org/df0b/05d8985846e694cda62d41a04e7c85090fa6.pdf

rn.seed(12345)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(3)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf.gpu_options.allow_growth = True
tf.set_random_seed(1234)
classes = 2
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(1234)


def build_model(X_train, row, cell):
    def square(x):
        return K.square(x)

    def log(x):
        return K.log(K.clip(x, min_value=1e-7, max_value=10000))
    
#     inpt = Input((3, X_train.shape[1], X_train.shape[2]))
    model = Sequential()
    model.add(Conv2D(40, kernel_size=(25, 1), data_format='channels_last',
                            input_shape=(X_train.shape[1], X_train.shape[2], 1)))
    model.add(Conv2D(40 , kernel_size=(X_train.shape[2], 1), use_bias=False, data_format='channels_last'))
    model.add(BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1))
    model.add(Activation(square))
    model.add(AveragePooling2D(pool_size=(75, 1), strides=(1, 15), data_format='channels_last'))
    model.add(Activation(log))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(3, kernel_constraint=max_norm(0.5)))
    model.add(Activation('softmax'))
    opt = optimizers.adam(lr=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
    plot_model(model, to_file='model.png', show_shapes=True)
    model.summary()
    return model


noLSTMOutputs = [32, 64, 128, 256]
if __name__ == '__main__':
    X_train, y_train, X_test, y_test, train_labels, test_labels = build_inputs(False, 330)
    X_train = X_train.reshape(-1, X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(-1, X_test.shape[1], X_test.shape[2], 1)
    epochs = 50  # 21
#     for q in range(1, 7):
#         for tt in range(1, 20):
    model = build_model(X_train, 0, 0)
    name = "{}-{}".format(0, 0)
    early_stop = EarlyStopping(monitor='val_acc', min_delta=0.1, patience=2,
                                    verbose=2, mode='auto')
    csv_logger = CSVLogger('res/training.csv', append=True, separator=',')
    history_callback = model.fit(X_train, y_train, epochs=epochs, batch_size=500,
        validation_split=0.2, callbacks=[csv_logger, early_stop])
    pred = model.predict(X_test)
    compute_accuracy(name, pred, test_labels, history_callback)
    evalRes(pred, test_labels, y_test, name)
