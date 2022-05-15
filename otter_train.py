""" Minimal script to build and train a single CNN with Keras
    and Tensorflow (v1.12).

    @F. Comitani 2018-2022
"""

import os
import argparse
import pickle

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import *


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train a single OTTER CNN')
    parser.add_argument('--data',  '-d', 
                    type    =  str,
                    help    =  'Path to input expression table file, pandas dataframe with samples as rows and features as columns')
    parser.add_argument('--labels',  '-l', 
                    type    =  str,
                    help    =  'Path to input labels file, one-hot-encoded pandas dataframes with samples as rows and classes as columns')
    parser.add_argument('--hparam',  '-hp', 
                    type    =  str,
                    help    =  'Path to hyperparameters file, json format')
    parser.add_argument('--lowvar',  '-lv', 
                    type    =  float,
                    help    =  'Cumulative variance threshold for low variance features removal, if 1 keep all features, default .9 keep all'+\
                                    'features reaching up to .9 of cumulative variance.',
                    default =  .9)
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
                    help    =  'Test fraction of training/test split, if 0 do not split, default .2',
                    default =  0)
    parser.add_argument('--outpath',  '-o', 
                    type    =  str,
                    help    =  'Path to output folder',
                    default =  './')

    args = parser.parse_args()

    
    """ Set constant variables. """

    epochs    = args.epochs
    batchsize = args.batchsize
    patience  = args.patience
    low_var   = args.lowvar
    split     = args.split
    outpath   = args.outpath

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    """ Load data. """

    print("Loading data...", end="")

    df     = pd.read_hdf(args.data)
    labels = pd.read_hdf(args.labels)
    labels = labels.loc[df.index]

    """ Remove cases with no labels from training. """

    labels = labels.loc[~(labels==0).all(axis=1)]
    df     = df.loc[labels.index]

    print('done!')

    print('Input data size: {:d}x{:d}'.format(*df.shape))
    print('Number of target labels: {:d}'.format(labels.shape[1]))

    """ Remove low variance features. """

    if low_var < 1:
        print("Removing low variance features...", end="")
        
        df = low_var_drop(df,thresh=low_var)

        print('done!')

        print('New input data size: {:d}x{:d}'.format(*df.shape))

    features = df.columns

    """ Fit and save a standard scaler. """

    print("Scaling...", end="")

    s_scale = StandardScaler()
    df      = s_scale.fit_transform(df)

    with open(os.path.join(outpath, 's_scaler.pkl'), 'wb') as handle:
        pickle.dump((s_scale,features), handle, protocol=-1)

    x_train   = df.reshape(df.shape[0], df.shape[1],1).astype('float32')
    y_train   = labels.values

    print('done!')

    valdata   = None
    if split > 0:

        print("Splitting train and test...", end="")

        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=split)
        valdata = (x_test, y_test)

        print('done!')

    traindata = (x_train, y_train)


    """ Build and train a model. """

    print("Training model...", end="")

    hparam = load_json_hparam(args.hparam)
    hparam['params']['input_shape'] = (df.shape[1],1)
    hparam['params']['num_classes'] = labels.shape[1]

    model   = build(hparam['params'])
    history = train(model, traindata, valdata, epochs, batchsize, patience)
    
    print('done!')

    """ Save trained model to disk. """

    print("Saving files...", end="")

    with open(os.path.join(outpath, hparam['model_name']+'_history.pkl'), 'wb') as handle:
        pickle.dump(history, handle, protocol=-1)

    model.save(os.path.join(outpath, hparam['model_name']+'_trained.h5'))

    """ Plot change in performance scores during training. """

    plot_history(history,'acc', outpath)
    plot_history(history,'f1', outpath)
    plot_history(history,'loss', outpath)
   
    print('done!')
    
    print('\nAll Done!')
