#!/usr/bin/python
"""Script to train videos in batches"""

import numpy as np
import tensorflow as tf
import i3d
import pickle
from tqdm import tqdm
import os
import sys
from time import gmtime, strftime
import time
import random
from behavior_recognition.data_io.tfrecord_reader import get_video_label_tfrecords
from behavior_recognition.data_io.fetch_balanced_batch import *
from behavior_recognition.tools.tf_utils import *
from behavior_recognition.tools.video_utils import *
from behavior_recognition.tools.utils import *

_IMAGE_SIZE = 224
_NUM_CLASSES = 9

H5_ROOT = '/media/data_cifs/mice/mice_data_2018/labels'
_SAMPLE_VIDEO_FRAMES = 6
_SAMPLE_PATHS = {
    'rgb': 'data/v_CricketShot_g04_c01_rgb.npy',
    'flow': 'data/v_CricketShot_g04_c01_flow.npy',
}
_CHECKPOINT_DIRS = {
        'mice': 'ckpt_dir/'
        }
_CHECKPOINT_PATHS = {
    'mice': 'ckpt_dir/Mice_ACBM_FineTune_I3D_Tfrecords_0.0001_Adam_10_85000_2018_06_30_07_20_32.ckpt.meta',
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

CLASSES_MICE = ["drink", "eat", "groom", "hang", "sniff", "rear", "rest", "walk", "eathand"]

def train_batch_videos(n_train_batches, n_epochs,
                        input_mode='rgb', save_every=5000,
                        tfrecords_filename=None,print_every=10,
                        action_every=50, num_classes=9,
                        n_frames=16, batch_size=10,
                        early_stopping=5,learning_rate=1e-4):
    #TODO: Implement validation
    """Function to train videos in batches.
        :param n_train_batches: Number of training batches in one epoch
        :param n_epochs: Number of epochs to train the model
        :param val_accuracy_iter: Interval for checking validation accuracy
        :param video2label: Dictionary mapping video ids to action labels
        :param input_mode: One of 'rgb','flow','two_stream'
        :param save_every: Save checkpoint of weights every save_every iterations
        :param print_every: Print loss log every print_every iterations
        :param action_every: Print action predictions with their respective
                             ground truth every action_every iterations
        :param num_classes: Number of action classes
        :param n_frames: Number of frames to represent a video
        :param batch_size: Batch size for training"""
    correct_preds = 0.
    preds_labels_cnf = []
    #saver_mice = tf.train.Saver()
    #step = get_optimizer(loss,optim_key='adam',learning_rate=learning_rate)
    with tf.Session().as_default() as sess:
        filename_queue = tf.train.string_input_producer([tfrecords_filename],
                                                                  num_epochs=1)
        videos,labels,masks = get_video_label_tfrecords(filename_queue,batch_size,
                                                    subset='train',shuffle=False)
        one_hot = tf.one_hot(labels, depth=num_classes, dtype=tf.int32)
        predictions,loss,top_classes,input_video_ph,input_video_ph_norm,ground_truth,saver = get_preds_loss_tfrecords(ground_truth=one_hot,
                                                                                                input_fr_rgb_unnorm=videos,
                                                                                                input_mode=input_mode,
                                                                                                n_frames=n_frames,
                                                                                                batch_size=batch_size,
                                                                                                dropout_keep_prob=1.0)
        labels_tf = tf.argmax(ground_truth, axis=1)
        init_op = tf.group(tf.global_variables_initializer(),
                tf.local_variables_initializer())
        saver_mice = tf.train.Saver()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord,sess=sess)
        if input_mode=='rgb':
            n_iters = int((n_epochs*n_train_batches))
            saver = tf.train.import_meta_graph(_CHECKPOINT_PATHS['mice'])
            saver.restore(sess, tf.train.latest_checkpoint(_CHECKPOINT_DIRS['mice']))
            i=0
            try:
                while(True):
                    start = time.time()
                    curr_loss,top_class_batch,videos_batch,labels_batch,one_hot_batch= sess.run([loss,
                                                                                   top_classes,
                                                                                   input_video_ph,
                                                                                   labels_tf,
                                                                                   ground_truth,
                                                                                   ])
                    end = time.time()
                    print 'Time elapsed: ',end - start
                    start = end
                    correct_preds += list(top_class_batch==labels_batch).count(True)
                    text_preds = [L_POSSIBLE_BEHAVIORS[b] for b in top_class_batch]
                    text_labels = [L_POSSIBLE_BEHAVIORS[b] for b in labels_batch]
                    #play_minibatch(videos_batch, text_preds, text_labels)
                    train_acc = round(correct_preds/float((i+1)*batch_size),3)
                    if i%print_every==0:
                        print 'Iteration-%s Current validation loss: %s Current validation accuracy: %s'%((i+1),
                                                                                                    curr_loss,
                                                                                                    train_acc)
                    if i%action_every==0:
                        print_preds_labels(top_class_batch, labels_batch)
                    preds_labels_cnf.extend([(p,l) for p,l in zip(top_class_batch, labels_batch)])
                    #if i and i%1000 == 0:
                    #    pickle.dump(preds_labels_cnf, open('cnf/Latest_Preds_Labels_CNF_%s.p'%(i),'w'))
                    #    preds_labels_cnf = []
                    i += 1
            except tf.errors.OutOfRangeError:
                print "TfRecords weren't written beyond this point :( restart training from ckpt"
            finally:
                coord.request_stop()
            coord.join(threads)

if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    print "Working on GPU %s"%(os.environ["CUDA_VISIBLE_DEVICES"])
    batch_size = 16
    n_batches = compute_n_batch(H5_ROOT,
                                  batch_size,
                                  ratio=0.25)
    train_batch_videos(n_train_batches=n_batches,
                        n_epochs=1,# video2label=video2label,
                        tfrecords_filename=sys.argv[2],
                        batch_size=batch_size,
                        #val_tfrecords=None,
                        learning_rate=1e-4)
