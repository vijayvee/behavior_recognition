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

_CHECKPOINT_PATHS = {
    'mice': 'ckpt_dir/Mice_ACBM_FineTune_I3D_Tfrecords_0.0001_Adam_10_85000_2018_06_30_07_20_32.ckpt.meta',
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

CLASSES_MICE = ["drink", "eat", "groom", "hang", "sniff", "rear", "rest", "walk", "eathand"]

def infer_video(video_path,input_mode='rgb', print_every=10,
                action_every=50, num_classes=9,
                n_frames=16, batch_size=16,
                VIDEO_LENGTH=108000,
                ):
    #TODO: Implement validation
    """Function to train videos in batches.
        """
    i3d_preds = []
    with tf.Session().as_default() as sess:
        videos = tf.placeholder(tf.uint8, shape=[batch_size,
                                                    n_frames,
                                                    _IMAGE_SIZE,
                                                    _IMAGE_SIZE,
                                                    3])
        top_classes = get_preds_loss_inference(input_fr_rgb_unnorm=videos,
                                                input_mode=input_mode,
                                                n_frames=n_frames,
                                                batch_size=batch_size,
                                                dropout_keep_prob=1.0)
        init_op = tf.group(tf.global_variables_initializer(),
                            tf.local_variables_initializer())
        if show_output:
            out_vid = cv2.VideoWriter('%s_predictions_i3d.avi'%(video_name),
                            cv2.VideoWriter_fourcc('M','J','P','G'),
                            30,(480,360))
        saver_mice = tf.train.Saver()
        sess.run(init_op)
        if input_mode=='rgb':
            saver = tf.train.import_meta_graph(_CHECKPOINT_PATHS['mice'])
            saver.restore(sess,
                tf.train.latest_checkpoint(_CHECKPOINT_DIRS['mice']))
            start = time.time()
            for ii in range(16,VIDEO_LENGTH,n_frames):
                frames, shape = get_video_chunk_cv2(video_path,ii,
                                                    n_frames)
                feed_dict = {videos: frames}
                predictions = sess.run([top_classes],
                                        feed_dict=feed_dict)
                text_labels = [L_POSSIBLE_BEHAVIORS[pred] for pred in predictions]
                i3d_preds.extend(text_labels)
                if show_output:
                    for fr_ind in range(n_frames):
                        img = frames[fr_ind]
                        cv2.putText(img, (100,100), text_labels[fr_ind]
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (255,255,0),2)
                        out_vid.write(img)
            end = time.time()
            print 'Time elapsed: ',end - start
        to_h5(i3d_preds, video_path)
        if show_output:
            out_vid.release()
        return i3d_preds

if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    print "Working on GPU %s"%(os.environ["CUDA_VISIBLE_DEVICES"])
    batch_size = 16
    i3d_preds = infer_video(sample_video_path, VIDEO_LENGTH=100)
    import ipdb; ipdb.set_trace();
