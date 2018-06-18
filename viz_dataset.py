from video_utils import *
from create_shuffle_dict import load_label
from random import sample
import sys
import pickle

'''Script to visualize sample videos and behavior labels'''

H5_ROOT = sys.argv[1]
VIDEO_ROOT = sys.argv[2] #'/media/data_cifs/mice/small.pvd_40'

def viz_from_pkl(behav=None):
    behav2video = pickle.load(
                    open('pickles/Behavior2Video_small.pvd_40.p'))
    while(True):
        if behav is None:
            behav = np.random.choice(behav2video.keys(),1)[0]
        video_fn = sample(behav2video[behav].keys(),1)[0]
        ind = sample(behav2video[behav][video_fn],1)[0]
        video_fn = '%s/%s'%(VIDEO_ROOT,
                              video_fn)
        starting_frame = ind-16
        video_chunk, _ = get_video_chunk_cv2(video_fn,
                                            starting_frame=ind,
                                            n_frames=16)
        video_chunk = np.expand_dims(video_chunk,0)
        play_minibatch(video_chunk, [behav])

def find_h5(video_file, all_h5_files):
    '''Function to find corresponding h5 file
       for a video file with name video_file'''

    video_fn = video_file.split('/')[-1].split('.')[0]
    h5_fn = [i for i in all_h5_files if i.count(video_fn)>0]
    return h5_fn

def play_sample(h5_file, video_file, behav='eat'):
    '''Play a random sample video with label'''
    labels, counts = load_label(h5_file)
    if labels.count(behav) == 0:
        print 'No %s found'%(behav)
        return
    ind = labels.index(behav) #sample(range(len(labels)), 1)[0]
    starting_frame = ind - 16
    video_chunk, _ = get_video_chunk_cv2(video_file,
                                      starting_frame=starting_frame,
                                      n_frames=16
                                      )
    video_chunk = np.expand_dims(video_chunk,0)
    play_minibatch(video_chunk, [labels[ind]])

def main():
    all_h5_files = glob.glob('%s/*.h5'%(H5_ROOT))
    all_video_files = glob.glob('%s/*.mp4'%(VIDEO_ROOT))
    #viz_from_pkl('rear')
    all_h5_files = pickle.load(open(
                    'pickles/small.pvd_40_gt_filenames.p'))
    all_video_files = pickle.load(
                         open('pickles/small.pvd_40_video_filenames.p'))
    all_video_files.sort()
    all_h5_files.sort()
#    cont = raw_input('Continue?')
    while(True):
        vid = sample(all_video_files, 1)[0]
        h5_file = find_h5(vid, all_h5_files)
        print vid, h5_file[0]
        if len(h5_file) > 0:
            play_sample(video_file=vid,
                          h5_file=h5_file[0])
#        cont = raw_input('Continue?')

if __name__ == '__main__':
    main()
