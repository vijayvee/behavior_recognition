#!/usr/bin/python
"""File to read tfrecords and return labels,videos"""
import tensorflow as tf
import numpy as np
import sys
import os
from video_utils import *
from tqdm import tqdm

CLASSES_MICE = ["drink", "eat", "groom", "hang", "sniff", "rear", "rest", "walk", "eathand"]

L_POSSIBLE_BEHAVIORS = ["drink",
                        "eat",
                        "groom",
                        "hang",
                        "sniff",
                        "rear",
                        "rest",
                        "walk",
                        "eathand"]

def get_text_labels(labels):
    text_labels = []
    for i,ground_truth in enumerate(labels):
        print i,"Label: ",L_POSSIBLE_BEHAVIORS[ground_truth]
        text_labels.append(L_POSSIBLE_BEHAVIORS[ground_truth])
    return text_labels

def get_video_label_tfrecords(filename_queue,batch_size,
                                subset,shuffle=False):
    feature = {'{}/video'.format(subset): tf.FixedLenFeature([], tf.string),
              '{}/label'.format(subset): tf.FixedLenFeature([], tf.int64)}
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features=feature)
    label = tf.cast(features['{}/label'.format(subset)], tf.uint8)
    video_dec = tf.decode_raw(features['{}/video'.format(subset)],tf.uint8)
    #Reshape video data into the original shape
    video = tf.reshape(video_dec, [16, 480, 640, 3])
    # Creates batches by randomly shuffling tensors
    if shuffle:
        videos, labels = tf.train.shuffle_batch([video, label], seed=1234,
                                                  batch_size=16,
                                                  capacity=30,
                                                  num_threads=10,
                                                  min_after_dequeue=16)
        return videos,labels
    videos, labels = tf.train.batch([video, label],
                                      batch_size=16,
                                      capacity=30,
                                      num_threads=10)
    return videos, labels

def test_tfrecord_read(tfrecords_filename):
    with tf.Session().as_default() as sess:
        filename_queue = tf.train.string_input_producer([tfrecords_filename],
                                                          num_epochs=None)
        cont='y'
        videos,labels = get_video_label_tfrecords(filename_queue,1,
                                                    subset='train',
                                                    shuffle=False)
        init_op = tf.group(tf.global_variables_initializer(),
        tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord,sess=sess)
        for i in tqdm(range(1,100)):
            print 'Fetching batch'
            videos_batch,labels_batch = sess.run([videos,labels])
            print 'Fetched batch'
            text_labels = get_text_labels(labels_batch)
            print videos_batch.shape,labels_batch
            import ipdb; ipdb.set_trace()
            #invert_preprocessing(videos_batch,labels_batch,display=True)
            #cont = raw_input('One more batch?(y/n)')

def main():
    #os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[2]
    test_tfrecord_read(sys.argv[1])

if __name__=="__main__":
    main()
