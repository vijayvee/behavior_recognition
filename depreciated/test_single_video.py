#!/usr/bin/python
"""Loads a single video and returns action predictions"""

import numpy as np
import tensorflow as tf
from video_utils import *
import i3d

_IMAGE_SIZE = 224
_NUM_CLASSES = 400

_SAMPLE_VIDEO_FRAMES = 79
_SAMPLE_PATHS = {
    'rgb': 'data/v_CricketShot_g04_c01_rgb.npy',
    'flow': 'data/v_CricketShot_g04_c01_flow.npy',
}

_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

_LABEL_MAP_PATH = 'data/label_map.txt'
CLASSES_KIN = [x.strip() for x in open(_LABEL_MAP_PATH)]

def get_preds_tensor(input_mode='rgb',n_frames=79):
    """Function to get the predictions tensor, input placeholder and saver object
        :param input_mode: One of 'rgb','flow','two_stream'"""
    if input_mode == 'rgb':
        rgb_variable_map = {}
        input_fr_rgb = tf.placeholder(tf.float32,
                                        shape=[1, n_frames,
                                                _IMAGE_SIZE, _IMAGE_SIZE,
                                                3],
                                        name="Input_Video_Placeholder")
        with tf.variable_scope('RGB'):
            #Building I3D for RGB-only input
            rgb_model = i3d.InceptionI3d(_NUM_CLASSES,
                                            spatial_squeeze=True,
                                            final_endpoint='Logits')
            rgb_logits,_ = rgb_model(input_fr_rgb,
                                      is_training=False,
                                      dropout_keep_prob=1.0)

        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'RGB':
                rgb_variable_map[variable.name.replace(':0','')] = variable
        rgb_saver = tf.train.Saver(var_list = rgb_variable_map,
                                    reshape=True)
        model_predictions = tf.nn.softmax(rgb_logits)
        return model_predictions, input_fr_rgb, rgb_saver
    else:
        print "#TODO: Implement other input modes"

def predict_single_video(video_fn, n_frames):
    """Function to predict actions for the video given by video_fn
        video_fn: Filename of the video to predict for
        n_frames: Number of frames to use to represent the video"""
    video_frames_rgb, _shape = load_video_with_path_cv2(video_fn, n_frames)
    video_frames_rgb = np.expand_dims(video_frames_rgb,0)
    preds, input_video_ph, saver = get_preds_tensor(n_frames=n_frames)
    input_mode = 'rgb'
    with tf.Session() as sess:
        if input_mode == 'rgb':
            saver.restore(sess, _CHECKPOINT_PATHS['rgb'])
            predictions = sess.run([preds], feed_dict = {input_video_ph: video_frames_rgb})
            top_class = np.argmax(predictions)
        #TODO: Implement other input modes
        return CLASSES_KIN[top_class], predictions


def main():
    video_fn = '/media/data_cifs/cluster_projects/action_recognition/ActivityNet/Crawler/Kinetics/val/fixing hair/TkMVNYg1Nyc_000107_000117.mp4'
    n_frames = 79
    top_class,preds = predict_single_video(video_fn,n_frames)
    print top_class

if __name__=="__main__":
    main()
