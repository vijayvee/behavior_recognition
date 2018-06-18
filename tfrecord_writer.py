#!/usr/bin/python

"""Script to write tfrecords for action recognition - Kinetics"""

from random import shuffle
import glob
import cv2
import tensorflow as tf
import numpy as np
import sys
import os
from tqdm import tqdm
import imageio
from video_utils import *
from random import sample

L_POSSIBLE_BEHAVIORS = ["drink",
                        "eat",
                        "groom",
                        "hang",
                        "sniff",
                        "rear",
                        "rest",
                        "walk",
                        "eathand"]

def init(subset):
    # shuffle the addresses before saving
    shuffle_data = True
    #Load paths to videos organized in folders named by activities
    act_paths = glob.glob('/media/data_cifs/cluster_projects/action_recognition/ActivityNet/Crawler/Kinetics/%s/*'%(subset))
    video_paths = []
    _LABEL_MAP_PATH = 'data/label_map.txt'
    CLASSES_KIN = [x.strip() for x in open(_LABEL_MAP_PATH)]
    for act in act_paths:
        video_paths.extend(glob.glob(os.path.join(act,'*mp4')))
    # read addresses and labels from the 'train' folder
    #the second last element of split contains action name
    labels = [CLASSES_KIN.index(video.split('/')[-2]) for video in video_paths]

    # to shuffle data
    if shuffle_data:
        c = list(zip(video_paths, labels))
        shuffle(c)
        video_paths, labels = zip(*c)
    return video_paths,labels

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_tfrecords(data_path,video_paths,
                      action_labels,n_vids_per_batch,
                      subset):
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(data_path)
    video_count = 0
    for i in tqdm(range(len(video_paths)),desc='Writing tf records..'):
        # print how many videos are saved every 1000 videos
        if (i!=0 and (not i % n_vids_per_batch)):
            print 'Train data: {}/{}\nVideo type:{}'.format(i,
                                                              len(video_paths),
                                                              type(vid))
        # Load the video
        vid,_ = load_video_with_path_cv2(video_paths[i],n_frames=79)
        if type(vid)==int:
            #Video does not exist, load video returned -1
            print "No video {} exists {}".format(video_paths[i],vid)
            continue
        label = action_labels[i]
        # Create a feature
        feature = {'%s/label'%(subset): _int64_feature(label)}
        #for i in range(vid.shape[0]):
        feature['%s/video'%(subset)] = _bytes_feature(
                                            tf.compat.as_bytes(
                                                vid.tostring()))
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize to string and write on the file
        if example is not None:
            writer.write(example.SerializeToString())
            video_count += 1
        else:
	    print "Example is None"

    writer.close()
    sys.stdout.flush()
    return video_count

def main():
    subset='val'
    video_paths,labels = init(subset)
    n_videos_written = write_tfrecords(
                        '/media/data_cifs/cluster_projects/action_recognition/data/%s_2.tfrecords'%(subset),
                                                                                                    video_paths,
                                                                                                    labels,100,
                                                                                                    subset)
    print "{} videos written to tfrecords".format(n_videos_written)

if __name__=="__main__":
    main()
