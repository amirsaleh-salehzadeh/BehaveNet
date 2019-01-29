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
from DeepEEGSingleElectrode.input_preparation import build_inputs
from keras.callbacks import EarlyStopping, CSVLogger
from DeepEEG.evaluation import compute_accuracy, evalRes
from keras.utils.vis_utils import plot_model
from keras.constraints import max_norm
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.engine.input_layer import Input
from keras.engine.training import Model
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
    data_shape = (X_train.shape[1], X_train.shape[2], 1)
    input_main = Input((X_train.shape[1], X_train.shape[2], 1))
    block1 = Conv2D(25, (2, 1), data_format='channels_last',
             input_shape=data_shape)(input_main)


    block1 = Conv2D(25, (X_train.shape[2], 1))(block1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(2, 1), strides=(1, 2))(block1)
    block1 = Dropout(0.5)(block1)
  
    block2 = Conv2D(50, (5, 1))(block1)
    block2 = BatchNormalization(axis=1)(block2)
    block2 = Activation('elu')(block2)
    block2 = MaxPooling2D(pool_size=(2, 1), strides=(1, 2))(block2)
    block2 = Dropout(0.5)(block2)
    
    block3 = Conv2D(100, (5, 1))(block2)
    block3 = BatchNormalization(axis=1)(block3)
    block3 = Activation('elu')(block3)
    block3 = MaxPooling2D(pool_size=(2, 1), strides=(1, 2))(block3)
    block3 = Dropout(0.5)(block3)
    
    block4 = Conv2D(200, (5, 1),data_format='channels_last' )(block3)
    block4 = BatchNormalization(axis=1)(block4)
    block4 = Activation('elu')(block4)
    block4 = MaxPooling2D(pool_size=(2, 1), strides=(1, 2))(block4)
    block4 = Dropout(0.5)(block4)
    
    flatten = Flatten()(block4)
    
    dense   = Dense(3, kernel_constraint = max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)
    model = Model(inputs=input_main, outputs=softmax)
    opt = optimizers.adam(lr=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
    plot_model(model, to_file='model.png', show_shapes=True)
    
    return model

N_TIME_STEPS = 300
if __name__ == '__main__':
    X_train, y_train, X_test, y_test, train_labels, test_labels = build_inputs(False, N_TIME_STEPS)
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
