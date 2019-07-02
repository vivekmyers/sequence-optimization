from keras.layers import *
from keras.models import *
import numpy as np
from random import *
import tensorflow as tf
import keras.backend as K
from keras.callbacks import History
import models.cnn

class UncertainDropout(models.cnn.CNN):
    '''
    Uses dropout at test time to get an estimate of uncertainty.
    '''

    def make_net(self, alpha, opt, shape):
        self.add(Conv1D(64, kernel_size=7,
                           activation='relu',
                           input_shape=shape, kernel_regularizer=regularizers.l2(0.001)))
        self.add(Conv1D(64, kernel_size=5,
                           activation='relu', kernel_regularizer=regularizers.l2(0.001)))
        self.add(Conv1D(32, kernel_size=3,
                           activation='relu', kernel_regularizer=regularizers.l2(0.001)))
        self.add(Flatten())
        self.add(Dense(100, activation='relu'))
        self.add(Dropout(0.5, training=True))
        self.add(Dense(1, activation='sigmoid'))
        self.compile(loss='mse',
                    optimizer=opt,
                    metrics=['accuracy'])
        K.set_value(self.optimizer.lr, alpha)

    def predict(self, seqs):
        '''
        Return predictions as (mu, sigma) tuple.
        '''
        preds = np.array([np.squeeze(super().predict(np.array([self.encode(x) for x in seqs])) for i in range(self.itr)]))
        return preds.mean(axis=0), preds.std(axis=0)

    def __init__(self, itr, *args):
        super().__init__(*args)
        self.itr = itr
    

