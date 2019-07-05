import numpy as np
from random import *
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
stderr = sys.stderr
sys.stderr = open('/dev/null', 'w')
from keras.layers import *
from keras.models import *
import keras.backend as K
from keras.callbacks import History
sys.stderr = stderr


class CNN(Sequential):
    '''CNN with regularization for making simple sequence score predictions.'''

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
        self.add(Dropout(0.5))
        self.add(Dense(1, activation='sigmoid'))
        self.compile(loss='mse',
                    optimizer=opt,
                    metrics=['accuracy'])
        K.set_value(self.optimizer.lr, alpha)


    def fit(self, seqs, scores, val=([], []), epochs=1, verbose=2):
        x_val, y_val = val
        result = super().fit(x=np.array([self.encode(x) for x in seqs]), 
                        y=np.array(scores), validation_data=(np.array(
                            list(map(self.encode, x_val))), np.array(y_val)) if val[0] else None, 
                        verbose=verbose, epochs=epochs + self.current_epoch, 
                        initial_epoch=self.current_epoch, callbacks=[self.history])
        self.current_epoch += epochs
    
    def predict(self, seqs):
        return np.squeeze(super().predict(np.array([self.encode(x) for x in seqs])))
    
    def __call__(self, seqs):
        return self.predict(seqs)

    def __init__(self, encoder, alpha=1e-4, shape=()):
        super().__init__()
        self.encode = encoder
        self.history = History()
        self.current_epoch = 0
        self.alpha = alpha
        self.make_net(alpha, 'adam', shape)

