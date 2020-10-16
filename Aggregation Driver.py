# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 19:19:52 2020

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
test_size = int(0.6*fluo.shape[0])
test = fluo[test_size:]


dataX = [[acid_encode[a]/acid_count for a in seq] for seq in test['padded']]
X = np.reshape(dataX, (fluo.shape[0] - test_size, max_length, 1))
Y = np_utils.to_categorical([agg_encode[t] for t in test['aggregation']])

# Setup the model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(Y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# model.load_weights before compilation
weights = "GFP2 weights-19-1.1583.hdf5"
model.load_weights(weights)
model.compile(loss='categorical_crossentropy', optimizer='adam')

for seq in test['padded']:
    dataX2 = [acid_encode[a]/acid_count for a in seq]
    print(dataX2[0:5])
    X2 = np.reshape(dataX2, (1, X.shape[1], 1))
    predictions = model.predict(X2, verbose=1)
    model_guess = np.argmax(predictions)
    print("Model guess probability:", predictions[0][model_guess])
    print("Model guess:", agg_decode[model_guess])
    #print("Correct answer: ", test['aggregation'][seq])