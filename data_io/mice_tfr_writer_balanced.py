import pickle
import glob
import tensorflow as tf
import numpy as np
import sys
from behavior_recognition.tools.utils import *
import os
from tqdm import tqdm
from behavior_recognition.paths import *
from behavior_recognition.tools.tf_utils import *
from behavior_recognition.tools.tf_utils import _int64_feature, _bytes_feature
from behavior_recognition.data_io.fetch_balanced_batch import *
import h5py
from time import gmtime, strftime

H5_ROOT = label_root #'/media/data_cifs/mice/mice_data_2018/labels'
DATASET_NAME = sys.argv[1]
SUBSET = sys.argv[2]
OUTPUT_PATH = sys.argv[3]
RATIO = float(sys.argv[4])
currTime = strftime("%Y_%m_%d_%H_%M_%S", gmtime())
tfr_path = OUTPUT_PATH.split('/')
tfr_name = '%s_%s_%s_%s_%s'%(currTime,
                        str(RATIO),
                        SUBSET, DATASET_NAME,
                        tfr_path[-1])
OUTPUT_PATH = '/'.join(tfr_path[:-1] + [tfr_name])
print('Writing tfrecords:', OUTPUT_PATH)

def write_tfrecords(subset='train',
                      n_batches=None,
                      batch_size=16,
                      ratio=1.):
    """Function to write tfrecords.
        :param subset: One of 'train' or 'va'
                       indicating the dataset
                       being written
        :param n_batch: Number of batches of
                        dataset to be written
        :param batch_size: Size of mini-batch
                           to be written
        :param ratio: Proportion of dataset to
                      be written"""

    counts = {behav:0 for behav in L_POSSIBLE_BEHAVIORS}
    writer = tf.python_io.TFRecordWriter(OUTPUT_PATH)
    #Compute number of batches to write on tfrecords
    #to fully write our dataset
    n_batches = compute_n_batch(H5_ROOT,
                                  batch_size,
                                  ratio)
    #Dictionary mapping behavior to videos and
    #frame sequences in videos labeled as behavior
    b2v_pickle = '%s/Behavior2Video_%s_%s.p'%(
                                        PICKLES_PATH,
                                        DATASET_NAME,
                                        subset
                                        )
    behav2video = pickle.load(open(b2v_pickle))
    ########## Start writing tfrecords ##########
    for ii in tqdm(range(n_batches),
                    desc='Writing tf records..'):
        # Load the video
        video_chunks, labels = fetch_balanced_batch(behav2video, batch_size=batch_size)
        labels = [L_POSSIBLE_BEHAVIORS[l] for l in labels]
        for behav in labels:
            counts[behav] += 1
        #Convert labels to discrete category indices
        labels_int = [L_POSSIBLE_BEHAVIORS.index(label)
                          for label in labels]
        for i in tqdm(range(len(labels_int)),
                        desc='Writing single minibatch'):
            ########## Create tfrecord features ##########
            X = video_chunks[i,:,:,:,:]
            y = labels_int[i]
            #Check if video is valid and not all-zero
            mask = int(X.sum()>0)
            feature = {'%s/label'%(subset):
                        _int64_feature(y)}
            feature['%s/mask'%(subset)] = _int64_feature(mask)
            feature['%s/video'%(subset)] = _bytes_feature(
                                                    tf.compat.as_bytes(
                                                    X.tostring())
                                                    )
            ########## Create an example protocol buffer #
            example = tf.train.Example(
                                    features=tf.train.Features(
                                    feature=feature
                                    ))
            ## Serialize to string and write on the file #
            if example is not None:
                writer.write(example.SerializeToString())
            else:
                print "Example is None"
        if ii%500==0:
            sys.stdout.flush()
    writer.close()
    sys.stdout.flush()

def main():
    write_tfrecords(subset=SUBSET,
                      ratio=RATIO,
                      batch_size=8)

if __name__=='__main__':
    #Usage: python mice_tfr_writer_balanced.py <DATASET_NAME> <SUBSET> <TFRECORD NAME>
    #Example: python mice_tfr_writer_balanced.py all_mice train data/all_mice_train.tfrecords
    main()
