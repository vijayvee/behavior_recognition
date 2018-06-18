import glob
import h5py
import fnmatch
import os

mice_root = '/media/data_cifs/mice'

def load_matches(root, pattern):
    matches = []
    for root, dirnames, filenames in os.walk(root):
        for filename in fnmatch.filter(filenames,
                                        '%s'%(pattern)):
            matches.append(os.path.join(root, filename))
    return matches

def main():
    all_h5 = load_matches('%s/p1_data/mice/small.pvd_40'%(
                                                    mice_root),
                                                    '*ground*.h5'
                                                    )
    all_mp4 = load_matches(mice_root,
                            '*-0000.mp4')
    count=1
    matched_gt, matched_video = [], []
    for i, gt in enumerate(all_h5):
        for vid in all_mp4:
            vid_id = vid.split('/')[-1]\
                        .split('.mp4')[0]
            if gt.count(vid_id)>0 and vid.count('_data')==0:
                print '%s Ground truth: %s, \nVideo: %s\n'%(count, gt, vid)
                matched_gt.append(gt)
                matched_video.append(vid)
                count += 1

if __name__=='__main__':
    main()
