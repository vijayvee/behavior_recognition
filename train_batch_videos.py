#!/usr/bin/python
"""Script to train videos in batches"""

import numpy as np
import tensorflow as tf
from video_utils import *
from utils import *
import i3d
import pickle
from tqdm import tqdm
import os
import sys
import random
from tfrecord_reader import get_video_label_tfrecords
from test_batch_videos import evaluate_model
from time import gmtime, strftime
from fetch_balanced_batch import *
from tf_utils import *

_IMAGE_SIZE = 224
_NUM_CLASSES = 9

H5_ROOT = '/media/data_cifs/mice/mice_data_2018/labels'
_SAMPLE_VIDEO_FRAMES = 6
_SAMPLE_PATHS = {
    'rgb': 'data/v_CricketShot_g04_c01_rgb.npy',
    'flow': 'data/v_CricketShot_g04_c01_flow.npy',
}
_CHECKPOINT_DIRS = {
        'mice': 'ckpt_dir/Mice_ACBM_I3D_0.0001_adam_10_19000_2018_02_19_01_27_30.ckpt'
        }
_CHECKPOINT_PATHS = {
    'mice': 'ckpt_dir/Mice_ACBM_I3D_0.0001_adam_10_19000_2018_02_19_01_27_30.ckpt.meta',
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

CLASSES_MICE = ["drink", "eat", "groom", "hang", "sniff", "rear", "rest", "walk", "eathand"]

def train_batch_videos(n_train_batches, n_epochs,
                        input_mode='rgb', save_every=1000,
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
    #saver_mice = tf.train.Saver()
    #step = get_optimizer(loss,optim_key='adam',learning_rate=learning_rate)
    with tf.Session().as_default() as sess:
        filename_queue = tf.train.string_input_producer([tfrecords_filename],
                                                                  num_epochs=None)
        videos,labels,masks = get_video_label_tfrecords(filename_queue,batch_size,
                                                    subset='train',shuffle=True)
#        init_op = tf.group(tf.global_variables_initializer(),
#                            tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord,sess=sess)
        one_hot = tf.one_hot(labels, depth=num_classes, dtype=tf.int32)
        predictions,loss,top_classes,input_video_ph,input_video_ph_norm,ground_truth,saver = get_preds_loss_tfrecords(ground_truth=one_hot,
                                                                                                input_fr_rgb_unnorm=videos,
                                                                                                input_mode=input_mode,
                                                                                                n_frames=n_frames,
                                                                                                batch_size=batch_size,
                                                                                                dropout_keep_prob=0.8)
        labels_tf = tf.argmax(ground_truth, axis=1)
        step = get_optimizer(loss,optim_key='adam',learning_rate=learning_rate)
        init_op = tf.group(tf.global_variables_initializer(),
                tf.local_variables_initializer())
        saver_mice = tf.train.Saver()
        sess.run(init_op)
        if input_mode=='rgb':
            n_iters = int((n_epochs*n_train_batches))
            saver.restore(sess, _CHECKPOINT_PATHS['rgb'])
            try:
                for i in tqdm(range(0,n_iters),desc='Training I3D on mice train set...'):
                    curr_loss,top_class_batch,videos_batch,labels_batch,one_hot_batch,_= sess.run([loss,
                                                                                   top_classes,
                                                                                   input_video_ph,
                                                                                   labels_tf,
                                                                                   ground_truth,
                                                                                   step
                                                                                   ])
                    correct_preds += list(top_class_batch==labels_batch).count(True)
                    train_acc = round(correct_preds/float((i+1)*batch_size),3)
                    if i%print_every==0:
                        print 'Iteration-%s Current training loss: %s Current training accuracy: %s'%((i+1),
                                                                                                    curr_loss,
                                                                                                    train_acc)
                    if i%action_every==0:
                        print_preds_labels(top_class_batch, labels_batch)
                    if i%save_every==0:
                        curr_time = strftime("%Y_%m_%d_%H_%M_%S", gmtime())
                        saver_mice.save(sess,
                                './ckpt_dir/Mice_ACBM_FineTune_I3D_Tfrecords_%s_%s_%s_%s_%s.ckpt'%(
                                            learning_rate,'Adam',
                                            n_epochs,str(i),curr_time))
            except tf.errors.DataLossError:
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
                                  ratio=1.0)
    train_batch_videos(n_train_batches=n_batches,
                        n_epochs=10,# video2label=video2label,
                        tfrecords_filename=sys.argv[2],
                        batch_size=batch_size,
                        #val_tfrecords=None,
                        learning_rate=1e-4)
