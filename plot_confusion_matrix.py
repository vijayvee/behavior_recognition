#!/usr/bin/python
import glob
import os
import sys
import numpy as np
from behavior_recognition.tools.utils import plot_confusion_matrix
import pickle

def load_preds_labels(pickle_name):
    preds_labels = pickle.load(open(pickle_name))
    return preds_labels

def collect_preds_labels(pickle_path):
    all_pickles = glob.glob('%s/*.p'%(pickle_path))
    preds_labels = []
    for pkl in all_pickles:
        preds_labels.extend(load_preds_labels(pkl))
    return preds_labels

def confusion_matrix_master(pickle_path):
    preds_labels = collect_preds_labels(pickle_path)
    preds = [i[0] for i in preds_labels]
    labels = [i[1] for i in preds_labels]
    L_POSSIBLE_BEHAVIORS.sort()
    plot_confusion_matrix(preds, labels, L_POSSIBLE_BEHAVIORS,
                            title='Confusion matrix - Inception3D',
                            xlabel='I3D predictions',
                            ylabel='Ground truth',
                            out_file='cnf_matrix.png'
                            )

def main():
    pickle_path = sys.argv[1]
    confusion_matrix(pickle_path)
