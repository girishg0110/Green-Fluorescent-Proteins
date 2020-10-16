# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 13:34:48 2020

@author: giris
"""

import pandas as pd
import numpy as np
from keras.utils import np_utils

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint

# Read in the csv
fluo = pd.read_csv('FP-Database-CSV.csv')

# Preprocessing
    # (0) Delete rows without brightness values
    # (1) Encode each acid with a binary number
    # (2) Zero pad as necessary (_) to end
    # (3) EX: Encode each aggregation with a binary number
    # (4) Prepare training set: X and Y tensors -- check dimensions

# (0)
fluo = fluo.dropna(subset = ['aggregation'])

# (1) and max_length for (2)
acid_encode = set()
max_length = 0
for seq in fluo['Sequence']:
        acid_encode.update(seq)
        if (max_length < len(seq)):
            max_length = len(seq)
acid_count = len(acid_encode)

acid_decode = {i + 1 : a for i, a in enumerate(acid_encode)}        
acid_encode = {a : i + 1 for i, a in enumerate(acid_encode)}
acid_decode[0] = '_'
acid_encode['_'] = 0

# (2)
mod_seqs = {seq + (max_length-len(seq)) * "_" : i for i, seq in enumerate(fluo['Sequence'])}
fluo['padded'] = mod_seqs

# (3)
agg_encode = set(list(fluo['aggregation']))

agg_decode = {i : a for i, a in enumerate(agg_encode)}        
agg_encode = {a : i for i, a in enumerate(agg_encode)}

# (4)
# dataX = [np_utils.to_categorical([acid_encode[a] for a in seq]).tolist() for seq in train['padded']]
# X = np.reshape(dataX, (train_size, max_length, acid_count, 1))
train_size = int(0.6*fluo.shape[0])
train = fluo[:train_size]

dataX = [[acid_encode[a]/acid_count for a in seq] for seq in train['padded']]
X = np.reshape(dataX, (train_size, max_length, 1))
Y = np_utils.to_categorical([agg_encode[t] for t in train['aggregation']])

def train_gfp(X, Y):
    # Setup the model
    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(Y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    filepath = "GFP2 weights-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    model.fit(X, Y, epochs=20, batch_size=128, callbacks=callbacks_list)
    
# train_gfp(X, Y)