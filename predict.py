""" Minimal script to build and train a single CNN with Keras
    and Tensorflow (v1.12).

    @F. Comitani 2018-2022
"""

from __future__ import print_function
import pandas as pd
import numpy as np
import os
import argparse
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing

import tensorflow as tf
import keras
from keras import backend as K

#config = tf.ConfigProto(device_count={"CPU": 8})
#keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, BatchNormalization
from keras.optimizers import Adadelta
from keras.callbacks import EarlyStopping, ModelCheckpoint

from .metrics import *
from .utils import *


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train a single OTTER CNN')
    parser.add_argument('--data',  '-d', 
                    type    =  str,
                    help    =  'Path to input expression table file, samples as rows, features as columns')
    parser.add_argument('--labels',  '-l', 
                    type    =  str,
                    help    =  'Path to input labels file, one-hot-encoded')
    parser.add_argument('--hparam',  '-hp', 
                    type    =  str,
                    help    =  'Path to hyperparameters file, json format')
    parser.add_argument('--epochs',  '-e', 
                    type    =  int,
                    help    =  'Number of training epochs',
                    default =  50)
    parser.add_argument('--batchsize',  '-b', 
                    type    =  int,
                    help    =  'Batch size',
                    default =  64)
    parser.add_argument('--patience',  '-p', 
                    type    =  int,
                    help    =  'Early stopping patience',
                    default =  5)
    parser.add_argument('--split',  '-s', 
                    type    =  float,
                    help    =  'Test size (%) of training/test split, if 0 do not split',
                    default =  .2)
    parser.add_argument('--outpath',  '-o', 
                    type    =  str,
                    help    =  'Path to output folder',
                    default =  './')

    args = parser.parse_args()

    
    """ Set constant variables. """

    epochs    = args.epochs
    batchsize = args.batchsize
    patience  = args.patience
    outpath   = args.outpath

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    """ Load data. """

    path = os.getcwd()

    df     = pd.read_hdf(args.data)
    labels = pd.read_hdf(args.labels)
    labels = labels.loc[df.index]

    """ Remove cases with no labels from training. """

    labels = labels.loc[~(labels==0).all(axis=1)]
    df     = df.loc[labels.index]

    """ Fit and save a standard scaler. """

    sScale = StandardScaler()
    df     = Scale.fit_transform(df)

    with open(os.path.join(path, 'sScaler.pkl'), 'wb') as handle:
        pickle.dump(sScale, handle, protocol=-1)

    x_train   = df.reshape(df.shape[0], df.shape[1],1).astype('float32')
    y_train   = labels.values


    valdata   = None
    if split > 0:
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=args.split)
        valdata = (x_test, y_test)
    traindata = (x_train, y_train)

    """ Build and train a model. """

    hparam = load_json_hparam(args.hparam)
    hparam['input_shape'] = (df.values.shape[1],1)
    hparam['num_classes'] = labels.shape[1]

    model   = build(hparam['params'])
    history = train(model, traindata, valdata, epochs, batchsize, patience)
    
    """ Save trained model to disk. """

    save_model(model, os.path.join(outpath,hparam['model_name']+'_trained'))
    #or directly save with model.save 
    #model.save(os.path.join(outpath, name+'_trained.h5'))

    """ Plot change in performance scores during training. """

    plot_history(history,'acc',outpath)
    plot_history(history,'f1',outpath)
    plot_history(history,'loss',outpath)
   
    print('\nAll Done!')