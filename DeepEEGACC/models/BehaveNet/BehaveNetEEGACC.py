# import sys
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling1D, AveragePooling1D
from sklearn.ensemble.forest import RandomForestClassifier
from pandas.core.frame import DataFrame
from seaborn.matrix import heatmap
# sys.path.insert(0, "/home/cirl/Amir/Human-Activity-EEG-Accelerometer")
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten, Reshape
import time

import tensorflow as tf
import random as rn
from keras import backend as K, optimizers
from DeepEEGACC.input_preparation import build_inputs
from keras.callbacks import EarlyStopping, CSVLogger
from DeepEEG.evaluation import compute_accuracy, evalRes
from keras.utils.vis_utils import plot_model
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Conv2D, Conv1D, SeparableConv1D
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


# 16
def build_model(X_train, row, cell):
    model = Sequential()
    model.add(Conv1D(32, 10, strides=1, data_format='channels_last',
             input_shape=(330, 5)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Conv1D(64, 15, strides=4, data_format='channels_last'))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=16, strides=2, padding='valid'))
    model.add(Dense(512))
    model.add(Activation("tanh"))
    model.add(LSTM(128, activation='tanh', recurrent_activation='hard_sigmoid', \
                    use_bias=True, kernel_initializer='glorot_uniform', \
                    recurrent_initializer='orthogonal', \
                    unit_forget_bias=True, kernel_regularizer=None, \
                    recurrent_regularizer=None, \
                    bias_regularizer=None, activity_regularizer=None, \
                    kernel_constraint=None, recurrent_constraint=None, \
                    bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, \
                    implementation=1, return_sequences=True, return_state=False, \
                    go_backwards=False, stateful=False, unroll=False))
    model.add(Dropout(0.4))
    model.add(LSTM(128, activation='tanh', recurrent_activation='hard_sigmoid', \
                    use_bias=True, kernel_initializer='glorot_uniform', \
                    recurrent_initializer='orthogonal', \
                    unit_forget_bias=True, kernel_regularizer=None, \
                    recurrent_regularizer=None, \
                    bias_regularizer=None, activity_regularizer=None, \
                    kernel_constraint=None, recurrent_constraint=None, \
                    bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, \
                    implementation=1, return_sequences=False, return_state=False, \
                    go_backwards=False, stateful=False, unroll=False))
    model.add(Dropout(0.4))
    model.add(Dense(3, activation="softmax"))
    opt = optimizers.adam(lr=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
    plot_model(model, to_file='model.png', show_shapes=True)
    model.summary()
    return model


if __name__ == '__main__':
    X_train, y_train, X_test, y_test, train_labels, test_labels = build_inputs(False, 330)
    epochs = 50  # 21
#     rforest_checker = RandomForestClassifier(random_state = 0)
#     rforest_checker.fit(X_train, y_train)
#     importances_df = DataFrame(rforest_checker.feature_importances_, columns=['Feature_Importance'],
#                                   index=["AF7", "AF8", "X_Axis", "Y_Axis", "Z_Axis"])
#     importances_df.sort_values(by=['Feature_Importance'], ascending=False, inplace=True)
# #     colormap = plt.cm.viridis
#     
#     plt.figure(figsize=(12,12))
#     plt.title('Correlation between Features', y=1.05, size = 15)
#     tmp = np.corrcoef(X_train)
#     heatmap(tmp,
#                 linewidths=0.1, 
#                 vmax=1.0, 
#                 square=True, 
# #                 cmap=colormap, 
#                 linecolor='white', 
#                 annot=True)
#     print(importances_df)
    model = build_model(X_train, 0, 0)
    name = "{}-{}".format(0, 0)
    early_stop = EarlyStopping(monitor='val_acc', min_delta=0.1, patience=2, mode='auto')
    csv_logger = CSVLogger('res/training.csv', append=True, separator=',')
    history_callback = model.fit(X_train, y_train, epochs=epochs, batch_size=500,
        validation_split=0.2, verbose=1, callbacks=[csv_logger, early_stop])
    model.save_weights("model.h5")
    pred = model.predict(X_test)
    compute_accuracy(name, pred, test_labels, history_callback)
    evalRes(pred, test_labels, y_test, name)
