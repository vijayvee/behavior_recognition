import pickle
import glob
import h5py
import numpy as np
import tensorflow as tf

#Generic utils file for I/O, pickle read, and other numeric computations

def get_b2v(subset,DATASET_NAME):
    import pickle
    b2v_pickle = 'pickles/Behavior2Video_%s_%s.p'%(
                            DATASET_NAME,
                            subset
                            )
    behav2video = pickle.load(open(b2v_pickle))
    return behav2video

def compute_n_batch(H5_ROOT,
                       batch_size,
                       ratio=1.):
    '''Function to compute number of samples
       to load for writing tfrecords.
       :param H5_ROOT: Root directory containing
                       all h5 files
       :param batch_size: Size of each minibatch
       :param ratio: Proportion of dataset to be
                     written as tfrecords'''
    all_h5_files = glob.glob('%s/*.h5'%(H5_ROOT))
    nLabels = 0
    for h5_f in all_h5_files:
        labels, counts = load_label(h5_f)
        nLabels += len(labels)
    #To write only a proportion of the dataset
    nLabels = int(nLabels*ratio)
    return nLabels/batch_size

def load_label(label_path):
    f = h5py.File(label_path)
    labels = f['labels'].value
    behav, labels_count = np.unique(labels, return_counts=True)
    counts = {k:v for k,v in zip(behav,labels_count)}
    return list(labels), counts

def get_lists(subset,ratio):
    labels = '{}/{}_labels_norest.pkl'.format(data_root,subset)
    videos = '{}/{}_videos_norest.pkl'.format(data_root,subset)
    ind_s, ind_e = 0, int(len(videos)*ratio)
    subset_labels = pickle.load(open(labels))[ind_s:ind_e]
    subset_videos = pickle.load(open(videos))[ind_s:ind_e]
    return subset_videos, subset_labels
