import matplotlib.pyplot as plt
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


