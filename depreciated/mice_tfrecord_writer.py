#!/usr/bin/python

##################### DEPRECIATED CODE #####################
##################### INEFFICIENT BECAUSE ##################
##################### OF CLASS IMBALANCE ###################

"""Script to write tfrecords for mixed-mice-data
Separate script since chunks of video are not stored
in different files and filenames"""

from random import shuffle
import pickle
import h5py
import glob
import cv2
import tensorflow as tf
import numpy as np
import sys
import os
from tqdm import tqdm
import imageio
from tf_utils import *
from video_utils import *

def write_tfrecords(data_path,video_paths,action_labels,
                    n_vids_per_batch,subset,
                    n_frames_batch = 16,
                    n_frames_chunk = 512):
    """Function to write tfrecords.
        :param data_path: name of tfrecords file to write
        :param video_paths: list containing filenames of videos
        :param action_labels: list conatining filenames of
                              ground truth label h5 files"""
    counts = {behav:0 for behav in L_POSSIBLE_BEHAVIORS}
    writer = tf.python_io.TFRecordWriter(data_path)
    video_count = 0
    tot_num_chunks = 0
    for i in tqdm(range(len(video_paths)),
                    desc='Writing tf records..'):
        print '#'*80,'\n'
        video_name = video_paths[i].split('/')[-1]
        # Load the video
        label, counts_curr = load_label(action_labels[i])
        for behav,count in counts_curr.iteritems():
            if behav.lower() != 'none':
                counts[behav] += count

        ############### Read batches of video ###############

        for ii in tqdm(range(0, len(label),
                              n_frames_chunk),
                              desc='Reading batches of videos'):
            #load only as many frames for which labels are available
            j_range_max = min(len(label)-ii,n_frames_chunk)
            video,(n,h,w,c) = load_video_with_path_cv2_abs(
                                                    '%s/%s'%(
                                                    data_root,
                                                    video_paths[i],
                                                    dtype='uint8'),
                                                    starting_frame=ii,
                                                    n_frames=j_range_max)
            if type(video)==int:
                #Video does not exist, load video returned -1
                print "No video %s/%s exists %s"%(
                                                  data_root,
                                                  video_paths[i],
                                                  video
                                                  )
                continue
            if video.dtype != np.float32:
                video = video.astype(np.float32)
            #Incorporate shuffling within chunk
            curr_range = range(0,j_range_max-n_frames_batch)
            curr_num_chunks = len(curr_range)
            tot_num_chunks += curr_num_chunks
            shuffle(curr_range)
            for jj in tqdm(range(len(curr_range)),
                            desc='Writing frames for chunk %s of video %s'%(
                                                                ii/n_frames_chunk,
                                                                video_name
                                                                )):
                #Shuffled index j in current chunk
                j = curr_range[jj]
                vid = video[j:n_frames_batch+j]
                #Add ii to account for starting frame number
                label_action = label[ii+n_frames_batch+j-1]
                #Do not train with 'none' labels that are
                #present in the training h5 files
                if label_action.lower() == 'none':
                    continue
                label_int = L_POSSIBLE_BEHAVIORS.index(label_action)
                # Create a feature
                feature = {'%s/label'%(subset): _int64_feature(label_int)}
                feature['%s/video'%(subset)] = _bytes_feature(
                                                              tf.compat.as_bytes(
                                                              vid.tostring()
                                                              )
                                                              )
                # Create an example protocol buffer
                example = tf.train.Example(
                                        features=tf.train.Features(
                                        feature=feature
                                        ))
                # Serialize to string and write on the file
                if example is not None:
                    writer.write(example.SerializeToString())
                    video_count += 1
                else:
        	    print "Example is None"
	            sys.stdout.flush()
    writer.close()
    sys.stdout.flush()
    return tot_num_chunks

def main():
    subset = sys.argv[1]
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]
    videos, labels = get_lists(subset,0.3)
    print "Writing %s videos and labels"%(len(videos))
    tfr_fname = '%s_0_3_flush_shuffled_norest_f32_mixed_mice.tfrecords'%(subset)
    tot_num_chunks = write_tfrecords('data/%s'%(tfr_fname),
                                                videos,
                                                labels, 1,
                                                subset)
    print tot_num_chunks, "i chunks written"

if __name__=="__main__":
    main()
