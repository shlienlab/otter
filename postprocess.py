""" Postprocessing functions for OTTER predictions.
    Requires an anytree object with hierarchical 
    information in output from RACCOON.

    @F. Comitani 2018-2022
"""

import os
import argparse
import pickle

import numpy as np
import pandas as pd

from aroughcun.utils.trees import load_tree

def get_node_by_name(name, nodes):
    """ Given a RACCOON class name return the node 
        from the matching anynode tree structure.

    Args:
        name (string): class name.
        nodes (anytree tree): list of anytree nodes.

    Returns:
        node (anytree node): anytree node instance.
    """

    for node in nodes:
        if node.name == name:
            return node
            
def _recursive_post_prob(node, probs, changed):
    """ Adjust probabilities of a node and its children
        recursively.

    Args:
        node (anytree node): anytree node instance whose branch
            will be adjusted.
        probs (pandas dataframe): dataframe of probabilities to adjust.
        changed (list): a list keeping track of the visited nodes.

    Returns:
        probs (pandas dataframe): the adjusted probabilities.
    """
        
    for child in node.children:

        if child.name in changed:

            children_list = [nc.name for nc in child.children]

            ratio  = pd.Series(1, index=probs[child.name].index)
            newval = probs[child.name][probs[children_list].max(axis=1)>probs[child.name]]
            ratio[probs[children_list].max(axis=1)>probs[child.name]] =\
                newval.div(probs[children_list][probs[children_list].max(axis=1)>probs[child.name]].max(axis=1))

            probs[children_list] =\
                probs[children_list].mul(ratio,axis=0)

            probs = _recursive_post_prob(child, probs, changed)
            
    return probs

def recenter_midpoint(x, cutoff):
    """ Adjust a probability value recentering its midpoint 
        to the cutoff value.

    Args:
        x (float): original probability.
        cutoff (float): new midpoint value.

    Returns:
        (float): adjusted probability value after recentering.
    """

    norm = cutoff

    if x > cutoff:
        norm = 1-norm

    return np.max([np.min([(x-cutoff)*.5/norm+.5,1]),0])


def adjust_probabilities(probs, nodes=None, calibration_weights=None):
    """ Adjust probabilities obtained with OTTER, according to a given
        hierarchy and/or recalibraing them according to provided
        Youden indices.

    Args:
        probs (pandas dataframe): dataframe of probabilities to adjust.
        nodes (anytree tree): list of anytree nodes with information
            on the classes hierarchy.
        calibration_weights (array-like): an array with weights
            for calibration (e.g. Youden indices for each class).

    Returns:
        probs (pandas dataframe): the adjusted probabilities dataframe.
    """

    """ Recalibrate probabilities with given thresholds. """

    
    if calibration_weights is not None:
        
        colnames = probs.columns
        vec_rm = np.vectorize(recenter_midpoint)
        probs  = probs.apply(vec_rm, cutoff=calibration_weights, result_type='expand', axis=1)
        probs.columns = colnames

    """ Adjust the probability between parents and children by averaging recursively. """

    #if p>parent p
    #the max p will be reduced to its mean with the parent, and similarly the parent and siblings.
    #then iteratively adjust children if further necessary

    #safeguard against division by zero
    probs = probs.replace(0,1e-10)
    
    if nodes is not None:

        changed=[]
        for i in probs.columns[::-1]:

            node = get_node_by_name(str(i), nodes)

            if len(node.children)>0:
                
                children_list = [child.name for child in node.children]

                if any(probs[children_list].max(axis=1)>probs[i]):
                    changed.append(node)

                    summa=(probs[i][probs[children_list].max(axis=1)>probs[i]]\
                        +probs[children_list][probs[children_list].max(axis=1)>probs[i]].max(axis=1))

                    ratio=pd.Series(1, index=probs[i].index)
                    ratio[probs[children_list].max(axis=1)>probs[i]] =\
                        summa.div(2*probs[children_list][probs[children_list].max(axis=1)>probs[i]].max(axis=1))

                    probs[children_list] =\
                        probs[children_list].mul(ratio,axis=0)

                    ratio[probs[children_list].max(axis=1)>probs[i]] =\
                        summa.div(2*probs[i][probs[children_list].max(axis=1)>probs[i]])

                    probs[i] =\
                        probs[i].mul(ratio,axis=0)

                    probs = _recursive_post_prob(node, probs, changed)

    #no need to keep anything this small
    probs = probs.applymap(lambda x: 0 if x<=1e-5 else x)
        
    return probs


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train a single OTTER CNN')
    parser.add_argument('--pred',  '-p', 
                    type    =  str,
                    help    =  'Path to input prediction probability file in output from otter_predict.py,'+\
                        'a pandas dataframe with samples as rows and features as columns')
    parser.add_argument('--nodes',  '-n', 
                    type    =  str,
                    help    =  'Path to hierarchical tree file in output from RACCOON, json format.'+\
                        'if not provided the probabilities will not be adjusted.',
                    default = None)
    parser.add_argument('--calibration',  '-c', 
                    type    =  str,
                    help    =  'Path to calibration weights file (e.g. Youden index), pickle format with list or np.array.'+\
                        'if not provided, calibration will not be applied.',
                    default =  None)
    parser.add_argument('--outpath',  '-o', 
                    type    =  str,
                    help    =  'Path to output folder',
                    default =  './')

    args = parser.parse_args()

    """ Set constant variables. """

    pred_name    = args.pred
    nodes        = args.nodes
    calibration  = args.calibration
    outpath      = args.outpath

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    """ Load data. """

    print("Loading data...")

    pred     = pd.read_hdf(pred_name)

    #This shouldn't be necessary, but if the columns in your prediction
    #df are int and they are str in the tree json convert them by uncommenting
    #the following line
    pred.columns = [str(x) for x in pred.columns]

    print('Input data size: {:d}x{:d}'.format(*pred.shape))

    if nodes is not None:

        nodes = load_tree(nodes)
        print('Hierarchical structure has been provided, the probabilities will be adjusted accordingly.')

    if calibration is not None:
        with open(calibration, 'rb') as handle:
            calibration = pickle.load(handle)
        print('Calibration weights have been provided, the probabilities will be adjusted accordingly.')

    print('done!')


    """ Adjust probabilities. """

    print("Adjusting probabilities...", end="")

    pred = adjust_probabilities(pred, nodes, calibration)
    
    print('done!')

    """ Save new probabilities to disk. """

    print("Saving file...", end="")

    pred.to_hdf(os.path.join(outpath, pred_name.split('/')[-1][:-3]+'_adjusted.h5'), key='df')
   
    print('done!')

