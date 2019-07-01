from keras.layers import *
from keras.models import *
import numpy as np
from random import *
import tensorflow as tf
import keras.backend as KB
from keras.callbacks import History

class CNN(Sequential):
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
        KB.set_value(self.optimizer.lr, alpha)
    
    def encode(self, seq):
        if seq in self.cache:
            arr = self.cache[seq]
        else:
            arr = np.zeros([len(seq), 4])
            arr[(np.arange(len(seq)),
                 ['ATCG'.index(c) for c in seq])] = 1
            self.cache[seq] = arr
        return arr

    def fit(self, seqs, scores, val=([], []), epochs=1, verbose=2):
        x_val, y_val = val
        result = super().fit(x=np.array([self.encode(x) for x in seqs]), 
                        y=np.array(scores), validation_data=(np.array(
                            list(map(self.encode, x_val))), np.array(y_val)), 
                        verbose=verbose, epochs=epochs + self.current_epoch, 
                        initial_epoch=self.current_epoch, callbacks=[self.history])
        self.current_epoch += epochs
    
    def predict(self, seqs):
        return np.squeeze(super().predict(np.array([self.encode(x) for x in seqs])))
    
    def __call__(self, seqs):
        return self.predict(seqs)

    def __init__(self, alpha=1e-4, shape=()):
        super().__init__()
        self.history = History()
        self.cache = {}
        self.current_epoch = 0
        self.alpha = alpha
        self.make_net(alpha, 'adam', shape)
