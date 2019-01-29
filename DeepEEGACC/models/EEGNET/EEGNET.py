import sys
sys.path.insert(0, "/home/cirl/Amir/Human-Activity-EEG-Accelerometer")
import numpy as np
import os
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten,\
    SpatialDropout2D
import time

import tensorflow as tf
import random as rn
from keras import backend as K, optimizers
from DeepEEGACC.input_preparation import build_inputs
from keras.callbacks import EarlyStopping, CSVLogger
from DeepEEG.evaluation import compute_accuracy, evalRes
from keras.utils.vis_utils import plot_model
from keras.constraints import max_norm
from keras.layers.convolutional import Conv2D, DepthwiseConv2D, SeparableConv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import AveragePooling2D
from keras.layers import Input
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
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(1234)


def EEGNet(nb_classes = 3, Chans = 5, Samples = 330, 
             dropoutRate = 0.25, kernLength = 150, F1 = 2, 
             D = 2, F2 = 8, dropoutType = 'Dropout'):
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = Input(shape = (1, Samples, Chans))
    

    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                   input_shape = (1, Samples, Chans),
                                   use_bias = False)(input1)
    block1       = BatchNormalization(axis = 1)(block1)
    block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.) , data_format='channels_first')(block1)
    block1       = BatchNormalization(axis = 1)(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 2), data_format='channels_first')(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F2, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization(axis = 1)(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 1))(block2)
    block2       = dropoutType(dropoutRate)(block2)
        
    flatten      = Flatten(name = 'flatten')(block2)
    
    dense        = Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(0.25))(flatten)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)


noLSTMOutputs = [32, 64, 128, 256]
if __name__ == '__main__':
    X_train, y_train, X_test, y_test, train_labels, test_labels = build_inputs(False, 330)
    X_train = X_train.reshape(-1, 1, X_train.shape[1], X_train.shape[2])
    X_test = X_test.reshape(-1, 1, X_test.shape[1], X_test.shape[2])
    epochs = 50  # 21
#     for q in range(1, 7):
#         for tt in range(1, 20):
#     model = build_model(X_train, 0, 0)
    model = EEGNet()
    opt = optimizers.adam(lr=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
    plot_model(model, to_file='model.png', show_shapes=True)
    model.summary()
    name = "{}-{}".format(0, 0)
    early_stop = EarlyStopping(monitor='val_acc', min_delta=0.1, patience=2,
                                    verbose=2, mode='auto')
    csv_logger = CSVLogger('res/training.csv', append=True, separator=',')
    history_callback = model.fit(X_train, y_train, epochs=epochs, batch_size=500,
        validation_split=0.2, callbacks=[csv_logger, early_stop])
    pred = model.predict(X_test)
    compute_accuracy(name, pred, test_labels, history_callback)
    evalRes(pred, test_labels, y_test, name)
