#!/usr/bin/python
import imageio
import numpy as np
import cv2
from random import sample
import pandas as pd
import matplotlib.pyplot as plt
import glob
from time import sleep
import tensorflow as tf
import h5py

"""Script for video related utilities"""

BATCH_SIZE = 10
N_FRAMES = 79
IMAGE_SIZE = 224
NUM_CLASSES=400
KINETICS_ROOT = '/media/data_cifs/cluster_projects/action_recognition/ActivityNet/Crawler/Kinetics'
subset = 'train'
VIDEOS_ROOT = KINETICS_ROOT + '/' + subset
_LABEL_MAP_PATH = 'data/label_map.txt'
CLASSES_KIN = [x.strip() for x in open(_LABEL_MAP_PATH)]
CLASSES_MICE = ["drink", "eat", "groom", "hang", "sniff", "rear", "rest", "walk", "eathand"]
video2label = {}

def resize_tf(arr, IMAGE_SIZE=224):
    old_h, old_w = (tf.constant(arr.shape[2].value, dtype=tf.float32),
                    tf.constant(arr.shape[3].value, dtype=tf.float32))
    ratio = tf.divide(IMAGE_SIZE, old_w)
    new_h, new_w = tf.multiply(old_h, ratio), tf.multiply(old_w,ratio)
    new_h, new_w = tf.cast(new_h, tf.int32), tf.cast(new_w, tf.int32)
    old_h = tf.cast(old_h, tf.int32)
    pad_h = IMAGE_SIZE - new_h
    pad_bottom, pad_top = pad_h // 2, pad_h - (pad_h//2)
    arr_rsz = tf.image.resize_images(arr, (new_h, new_w),
                                        align_corners=True)
    arr_rsz = tf.cast(arr_rsz, tf.uint8)
    padded = tf.pad(arr_rsz, tf.Variable([[0,0],[0,0],
                                            [pad_top, pad_bottom],
                                            [0,0],[0,0]]))
    padded = tf.cast(padded, tf.uint8)
    return padded

def load_video_with_path_cv2(video_path, n_frames):
    """ Fuction to read a video, select a certain select number of
        frames, normalize and return the array of videos
    :param video_path: Path to the video that has to be loaded
    :param n_frames: Number of frames used to represent a video"""
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened()==False:
        return -1,-1
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ind_frames = map(int,np.linspace(0,video_length-1,n_frames))
    frameCount, index = 0,0
    vid = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frameCount += 1
            frame = cv2.resize(frame,(224,224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32)
            vid.append(frame)
        else:
            break
    curr_frames = [vid[i] for i in ind_frames]
    curr_frames = np.array(curr_frames)
    norm_frames = (curr_frames/127.5) - 1.
    norm_frames = norm_frames.astype(np.float32)
    return norm_frames,norm_frames.shape

def invert_preprocessing(norm_frames, labels = [], display=False):
    """ Function to invert the preprocessing performed before writing tfrecords.
    :param norm_frames: Array of frames that are in normalized form"""
    curr_frames = (norm_frames + 1.) * 127.5
    print curr_frames.shape
    if display:
        for i in range(len(curr_frames)):
            if i!=0 and labels[i] == labels[i-1]:
                continue
            curr_vid = curr_frames[i,:,:,:,:]
            im = curr_vid[0,:,:,:]
            print im.shape
            show = plt.imshow(im)
            if len(labels)>0:
                print CLASSES_MICE[labels[i]]
            for ii in range(len(curr_vid)):
                im = curr_vid[ii,:,:,:]
                show.set_data(im)
                plt.pause(1./30)
            #plt.pause(1)
        plt.show()
    return curr_frames

def play_minibatch(frames, labels = []):
    '''Function to play as video, a sequence
       of frames from a minibatch
       :param frames: Array of a minibatch
                      of frames to play as
                      a video
       :param labels: If not empty, display
                      each behavior while
                      playing the video'''
    for i in range(len(frames)):
        curr_vid = frames[i,:,:,:,:]
        im = curr_vid[0,:,:,:]
        print im.shape, im.dtype
        show = plt.imshow(im)
        if len(labels)>0:
            print labels[i]
        for ii in range(len(curr_vid)):
            im = curr_vid[ii,:,:,:]
            show.set_data(im)
            plt.pause(1./60)
        plt.pause(1./10)
    plt.show()


def get_video_capture(video_path, starting_frame):
    '''Function to load a cv2 video capture object
       :param video_path: Path of video to capture
       :param starting_frame: Starting frame from
                              which capture begins'''
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened()==False:
        return -1
    if type(starting_frame) == list:
        starting_frame = starting_frame[0]
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert video_length > starting_frame
    cap.set(1,starting_frame)
    return cap

def resize_frame(frame, IMAGE_SIZE=224):
    old_h, old_w = frame.shape[0], frame.shape[1]
    ratio = float(IMAGE_SIZE)/old_w
    new_h, new_w = int(old_h*ratio), int(old_w*ratio)
    frame = cv2.resize(frame, (int(new_w), int(new_h)))
    pad_h = IMAGE_SIZE - new_h
    pad_bottom, pad_top = pad_h // 2, pad_h - (pad_h // 2)
    frame = cv2.copyMakeBorder(frame,int(pad_top),int(pad_bottom),0,0,
                                cv2.BORDER_CONSTANT,value=0)
    return frame

def get_video_chunk_cv2(video_path, starting_frame,
                          n_frames, IMAGE_SIZE=224,
                          normalize=False,
                          dtype=np.uint8):
    """Fuction to read a video, convert all read
        frames into an array, normalize and return
        the array of videos
        :param video_path: Path to the video that
                           has to be loaded
        :param starting_frame: Frame from which
                               video capture starts
        :param n_frames: Number of frames used to
                         represent a video
        :param normalize: Boolean flag indicating
                          preprocessing/normalization
                          of the video being captured
        :param dtype: dtype to which the captured
                      video is cast. Must be a numpy
                      dtype"""

    frameCount, index = 0,0
    vid = []
    cap = get_video_capture(video_path,
                             starting_frame)
    if type(cap) == int:
        return -1,(-1,-1,-1,-1)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frameCount += 1
            frame = resize_frame(frame, IMAGE_SIZE)
            frame = frame.astype(dtype)
            vid.append(frame)
            if frameCount == n_frames:
                break
        else:
            break
    assert len(vid) == n_frames
    curr_frames = np.array(vid)
    if normalize:
        curr_frames = (curr_frames/127.5) - 1.
    curr_frames = curr_frames.astype(dtype)
    return curr_frames,curr_frames.shape

def print_preds_labels(preds,labels):
    """Function to print activity predictions and ground truth next to each other in words.
    :param preds: List of behavior predictions for a mini batch
    :param labels: List of behavior ground truth for the same mini batch"""
    for i,(prediction,ground_truth) in enumerate(zip(preds,labels)):
        print i,"Prediction: ",CLASSES_MICE[prediction],"Label: ",CLASSES_MICE[ground_truth]
    print list(preds==labels).count(True), "correct predictions"

def load_video_with_path(video_path, n_frames):
    """ Fuction to read a video, select a certain select number of
        frames, normalize and return the array of videos
    :param video_path: Path to the video that has to be loaded
    :param n_frames: Number of frames used to represent a video"""
    #imageio gives problems, seems unstable to read videos

    vid = imageio.get_reader(video_path,'ffmpeg')
    ind_frames = map(int,np.linspace(0,vid.get_length()-1,n_frames))
    curr_frames = [vid.get_data(i) for i in ind_frames]
    import ipdb; ipdb.set_trace()
    resized_frames = [cv2.resize(i,(224,224)) for i in curr_frames]
    curr_frames = np.array(resized_frames)
    #norm_frames = (curr_frames/127.5) - 1.
    return norm_frames,norm_frames.shape

def read_video(video_id,label,n_frames,subset):
    """Function to read a single video given by video_fn and return
        n_frames equally spaced frames from the video
        video_fn: Filename of the video to read
        n_frames: Number of frames to read in the video"""

    VIDEOS_ROOT = KINETICS_ROOT + '/' + subset
    video_fn = glob.glob("%s/%s/%s*mp4"%(VIDEOS_ROOT,label,video_id))
    if video_fn == []:
        return np.zeros((224,224,3)),(224,224,3)
    vid = imageio.get_reader(video_fn[0],'ffmpeg')
    ind_frames = map(int,np.linspace(0,vid.get_length()-1,n_frames))
    curr_frames = [vid.get_data(i) for i in ind_frames]
    resized_frames = [cv2.resize(i,(224,224)) for i in curr_frames]
    curr_frames = np.array(resized_frames)
    norm_frames = (curr_frames/127.5) - 1.
    return norm_frames,norm_frames.shape

def get_video2label(subset):
    """Function to load a mapping from video id to label"""

    subset_csv = KINETICS_ROOT + '/kinetics_{}.csv'.format(subset)
    df = pd.read_csv(subset_csv)
    video_ids, labels = df.youtube_id, df.label
    video2label = {v:l for v,l in zip(video_ids,labels)}
    return video2label

def get_class_weights(LABELS_ROOT):
    import h5py
    import glob
    all_labels = glob.glob(LABELS_ROOT + '/*h5')
    for label in all_labels:
        f = h5py.File(label)
        labels = f['labels'].value
        label, count = np.unique(labels, return_counts=True)


def download_clip(video_identifier, output_dir,
                  start_time, end_time,
                  tmp_dir='/tmp/kinetics',
                  num_attempts=5,):
    """Trim video and store chunks.

    arguments:
    ---------
    video_identifier: str
        Path to the video stored on disk
    output_dir: str
        File path where the video will be stored.
    start_time: float
        Indicates the begining time in seconds from where the video
        will be trimmed.
    end_time: float
        Indicates the ending time in seconds of the trimmed video.
    """
    # Construct command to trim the videos (ffmpeg required).
    for i in range(0,3600):
        output_filename = '%s/%s_%s_%s.mp4'%(output_dir,
                                           video_identifier,
                                           str(i),
                                           str(i+1))
        command = ['ffmpeg',
                   '-i', '"%s"' % tmp_filename,
                   '-ss', str(i),
                   '-t', '1',
                   '-c:v', 'libx264', '-c:a', 'copy',
                   '-threads', '1',
                   '-loglevel', 'panic',
                   '"%s"' % output_filename]
        command = ' '.join(command)
        try:
            output = subprocess.check_output(command, shell=True,
                                             stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            return status, err.output

        # Check if the video was successfully saved.
        status = os.path.exists(output_filename)
        os.remove(tmp_filename)
    return status, 'Downloaded'

def get_video_batch(video2label,batch_size=BATCH_SIZE,
                      validation=False,val_ind=0,n_frames=N_FRAMES,
                      class_index=True):
    """Function to return a random batch of videos for train mode and a specific set of videos for val mode.
        :param batch_size: Specifies the size of the batch of videos to be returned
        :param validation: Flag to specify training mode (True for val phase)
        :param val_ind: Index of the batch of val videos to retrieve"""

    if validation:
        VIDEOS_ROOT = KINETICS_ROOT + '/val'
        curr_videos = video2label.keys()[val_ind:val_ind+batch_size]
        curr_labels = [video2label[v] for v in curr_videos]
        #Implement videos that are missing.
        curr_video_paths = [glob.glob("%s/%s/%s*mp4"%(VIDEOS_ROOT,
                                                        label,
                                                        video_id))[0]
                                                        for video_id,label in zip(
                                                                            curr_videos,curr_labels
                                                                                )]
        video_rgb_frames = [load_video_with_path_cv2(curr_vid,n_frames)[0]
                              for curr_vid in curr_video_paths]
        #video_rgb_frames = [read_video(video_id,label,n_frames,'val')[0] for video_id, label in zip(curr_videos,curr_labels)]
        if class_index:
            curr_labels = [CLASSES_KIN.index(action) for action in curr_labels]
        return np.array(video_rgb_frames),np.array(curr_labels)
    else:
        VIDEOS_ROOT = KINETICS_ROOT + '/train'
        curr_inds = sample(0,len(video2label)-1,batch_size)
        curr_videos = video2label.keys()[curr_inds]
        curr_labels = [video2label[v] for v in curr_videos]
        curr_video_paths = [glob.glob("%s/%s/%s*mp4"%(VIDEOS_ROOT,
                                                        label,
                                                        video_id))[0]
                                                        for video_id,label in zip(
                                                                curr_videos,curr_labels)]
        video_rgb_frames = [load_video_with_path_cv2(curr_vid,n_frames)[0] for curr_vid in curr_video_paths]
        if class_index:
            curr_labels = [CLASSES_KIN.index(action) for action in curr_labels]
        return np.array(video_rgb_frames),np.array(curr_labels)
