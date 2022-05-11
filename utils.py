""" Auxiliary functions for OTTER training and run.

    @F. Comitani 2018-2022
"""

import os
import json

import pandas as pd

import tensorflow as tf
import keras
from keras import backend as K

#config = tf.ConfigProto(device_count={"CPU": 8})
#keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, BatchNormalization
from keras.optimizers import Adadelta
from keras.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt

from metrics import *

def load_json_hparam(filename):
    """Load JSON from a path (directory + filename).
    
    Args:
        filename (str): name of the JSON file to load.
    
    """
    
    with open(filename, 'r') as f:
        return json.JSONDecoder().decode(
                f.read()
                )

def reset_weights(model):
    """ Reset model's weights. 

    Args:
        model (keras obj): model whose weights will be reset.
    """

    session = K.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)

def low_var_drop(df,thresh=0.99):
    """ Remove low variance features from a pandas dataframe.


    Args:
        df (pandas dataframe): the dataframe to transform.
        threhs (float): cumulative variance value threshold,
            features beyond this threshold will be removed.

    Return:
        (pandas dataframe): the transformed dataframe.
    """
    
    vVal=df.var(axis=0).values
    cs=pd.Series(vVal).sort_values(ascending=False).cumsum()
    remove=cs[cs>cs.values[-1]*thresh].index.values
    
    return df.drop(df.columns[remove],axis=1)

def build(hparam):
    """Build a model according to hyper-space parameters.

    Args:
        hparam (dict): dictionary containing hyperparameters
            values.

    Returns:
        model (Keras obj): Keras model.
    """ 

    print("Hyperparameters:")
    print(hparam)

    """ Build model sequentially based on provided hyperparameters. """
    model = Sequential()

    #First conv layer
    if hparam['first_conv'] is not None:
        model.add(Conv1D(32,kernel_size=int(hparam['first_conv']),
            activation='relu',
            input_shape=hparam['input_shape']))

    #Hidden conv layers
    n_nodes = int(64 * hparam['conv_hid_uni_mult'])
    for i in range(hparam['conv_pool_layers']):
        if hparam['first_conv'] is None:
            model.add(Conv1D(n_nodes, kernel_size=int(hparam['conv_kernel_size']), activation='relu', input_shape=hparam['input_shape']))
        else:
            model.add(Conv1D(n_nodes, kernel_size=int(hparam['conv_kernel_size']), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=4, stride=2))

    model.add(Flatten())

    #Hidden dense layers
    for i in range(hparam['dense_layers']):
        n_base=int(1024.0/(i*0.5+1))
        n_nodes = int(n_base * hparam['dense_hid_uni_mult'])
        model.add(Dense(n_nodes, activation='relu'))
        model.add(Dropout(0.6))

    #Final output layer
    model.add(Dense(hparam['num_classes'], activation='sigmoid'))

    #Compile the model
    model.compile(loss=keras.losses.binary_crossentropy,
            optimizer=Adadelta(),
            metrics=['accuracy', f1, precision, recall])

    return model

def train(model, traindata, valdata=None, epochs = 50, batchsize = 64, patience = 3):
    """Train a model.

    Args: 
        model (Keras obj): a Keras model.
        traindata (tuple numpy array): tuple containing training data and labels
        valdata (tuple numpy array): tuple containing validation data and labels, 
            if None do not run validation (default None).
        epochs (int): number of epochs (default 50).
        batchsize (int): batch size (default 64).
        patience (int): number of patience epochs for early stopping (default 3,
            ignored if valdata is None)
    
    Returns:
        fitting (Keras obj): the fitted model.
    """

    reset_weights(model)

    K.set_learning_phase(1)
    K.set_image_data_format('channels_last')

    """ Set early stopper. """

    callbacks = None
    if valdata is not None:
        earlyStop = EarlyStopping(monitor='val_loss', patience=patience, mode='auto', restore_best_weights=True)
        callbacks = [earlyStop]

    """ Train model. """
    fitting=model.fit(traindata[0], traindata[1],
                                batch_size=batchsize,
                                epochs=epochs,
                                shuffle=True,
                                verbose=1,
                                validation_data=valdata,
                                callbacks=callbacks).history

    """ Performance scores. """

    print("Training Results\n==========================================")
    print('Best Loss:', min(fitting['loss']))
    print('Best Accuracy:', max(fitting['acc']))
    print('Best F1:', max(fitting['f1']))
    print('Best Precision:', max(fitting['precision']))
    print('Best Recall:', max(fitting['recall']))

    if valdata is not None:
        print("Validation Results\n==========================================")
        print('Best Loss:', min(fitting['val_loss']))
        print('Best Accuracy:', max(fitting['val_acc']))
        print('Best F1:', max(fitting['val_f1']))
        print('Best Precision:', max(fitting['val_precision']))
        print('Best Recall:', max(fitting['val_recall']))

    return fitting


def save_model(model, name):
    """ Save a Keras model to a JSON file (architecture)
        and an hdf5 (weights) file.

    Args: 
        model (Keras obj): the keras model to save.
        name (str): the name of the file where the model
            will be saved.
    """
    
    """ Serialize the model to JSON. """

    model_json = model.to_json()
    with open(name+".json", "w") as json_file:
            json_file.write(model_json)
    
    """ Write the weights to disk. """
    model.save_weights(name+".h5")
    

def load_model(json_filename, weights_filename):
    
    """ Load a Keras model from a JSON file (architecture)
        and an hdf5 (weights) file.

    Args:
        name (str): name of the files to load.

    """

    from keras.models import model_from_json
        
    """ Load JSON and create model. """

    json_file = open(json_filename, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    """ Load trained weights into the new model. """

    loaded_model.load_weights(weights_filename)
    
def plot_history(history, metric, path='./'):
    """ Plot the change in a given metric 
        through both training and validation epochs.

    Args:
        history (dict): dictionary containing the values 
            to plot.
        metric (str): key for the history dictionary,
            the name of the metric to plot.
        path (str): path where the png plot will be saved.i

    """

    plt.plot(history[metric])
    legs=['train']

    if 'val_'+metric in history:
        plt.plot(history['val_'+metric])
        legs.append('val')
    
    plt.title('model '+metric)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(legs, loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(path, metric+'_training.png'), bbox_inches='tight', dpi=600)
    plt.clf()


if __name__ == "__main__":

    pass