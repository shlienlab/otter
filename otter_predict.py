""" Minimal script for inference with 
    an ensemble of CNN (OTTER).

    @F. Comitani 2018-2022
"""

import os
import sys
import argparse
import pickle

import numpy as np
import pandas as pd

from utils import *
from metrics import *


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Prediction script for OTTER CNN models')
    parser.add_argument('--data',  '-d', 
                    type    =  str,
                    help    =  'Path to input expression table file to which the inference will be applied,'+\
                        'pandas dataframe with samples as rows and features as columns')
    parser.add_argument('--models',  '-m', 
                    type    =  str,
                    help    =  'Path to folder with trained models file, all models must be in the same folder.'+\
                        'It should include the scaler produced by otter_train.py.'+\
                        'Architecture (json) and weights (h5) must have the same name, as in output of otter_train.py.')
    parser.add_argument('--outpath',  '-o', 
                    type    =  str,
                    help    =  'Path to output folder',
                    default =  './')

    args = parser.parse_args()

    """ Set constant variables. """

    modelpath   = args.models
    outpath   = args.outpath

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    """ Load data. """

    print("Loading data...", end="")

    df      = pd.read_hdf(args.data)
    indices = df.index

    print('done!')

    print('Input data size: {:d}x{:d}'.format(*df.shape))

    """ Select features and apply scaling. """

    print("Scaling...", end="")

    try:
        with open(os.path.join(modelpath, 's_scaler.pkl'), 'rb') as handle:
            s_scale, feats = pickle.load(handle)
    except FileNotFoundError:
        print("s_scaler.pkl not found!\nMake sure it is in the models folder.")
        sys.exit(1)

    try:
        df = s_scale.fit_transform(df[feats])
    except KeyError:
        print("Input features not found!\nMake sure the input format is correct")
        sys.exit(1)

    df = s_scale.transform(df)
    df = df.reshape(df.shape[0], df.shape[1],1).astype('float32')

    print('done!')

    """ Inference. """

    predictions = []

    for file in os.listdir(modelpath):
        if file.endswith('.h5'):
            try:
                net = keras.models.load_model(os.path.join(modelpath,file), 
                    custom_objects={'f1': f1, 'precision': precision, 'recall': recall}) 
                predictions.append(net.predict(df))
            except:
                print("Attempt at loading a model failed:", sys.exc_info()[0])

    if len(predictions)==0:
        print('Error: no models were loaded')
        sys.exit(1)

    predictions = pd.DataFrame(np.mean(predictions,axis=0), 
                    index=indices)

    """ Save results to disk. """

    print("Saving files...", end="")

    predictions.to_hdf(os.path.join(outpath, 'predictions.h5'), key='df')

    print('done!')
    
    print('\nAll Done!')
