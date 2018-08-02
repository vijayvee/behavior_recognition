BEHAV_REC_ROOT='/media/data_cifs/cluster_projects/behavior_recognition'
CKPT_DIR = '%s/ckpt_dir'%(BEHAV_REC_ROOT)
data_root = '/media/data_cifs/mice/mice_data_2018'
video_root = '{}/videos'.format(data_root)
label_root = '{}/labels'.format(data_root)
PICKLES_PATH = '%s/pickles'%(BEHAV_REC_ROOT)
H5_ROOT = '/media/data_cifs/mice/mice_data_2018/labels';
SAMPLE_VIDEO_FRAMES = 6
SAMPLE_PATHS = {
    'rgb': 'data/v_CricketShot_g04_c01_rgb.npy',
    'flow': 'data/v_CricketShot_g04_c01_flow.npy',
}
CHECKPOINT_DIRS = {
        'mice': 'ckpt_dir/'
        }
CHECKPOINT_PATHS = {
    'mice': 'ckpt_dir/Mice_ACBM_FineTune_I3D_Tfrecords_0.0001_Adam_10_85000_2018_06_30_07_20_32.ckpt.meta',
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}
CHECKPOINTS = {
    'mice': 'ckpt_dir/Mice_ACBM_FineTune_I3D_Tfrecords_0.0001_Adam_10_85000_2018_07_02_00_18_35.ckpt',
    }
