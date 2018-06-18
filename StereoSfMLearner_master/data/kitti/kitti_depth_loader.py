from __future__ import division
import numpy as np
from glob import glob
import os
import skimage.transform
# import sys
# sys.path.append('../../')
# from utils.misc import *

# Maximum disparity of gcnet for normalizing numpy float
MAX_DISPARITY = 192

class kitti_depth_loader(object):
    """ Class responsible for loading precalculated depths
    """
    def __init__(self,
                 dataset_dir,
                 img_height=128,
                 img_width=416,
                 seq_length=3):
        self.dataset_dir = dataset_dir
        self.img_height = img_height
        self.img_width = img_width
        self.seq_length = seq_length
        self.train_seqs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        self.test_seqs = [9, 10]

        self.collect_test_frames()
        self.collect_train_frames()

    def collect_test_frames(self):
        self.test_frames = []
        for seq in self.test_seqs:
            img_dir = os.path.join(self.dataset_dir, '%.2d' % seq)
            N = len(glob(img_dir + '/*.npy'))
            for n in range(N):
                self.test_frames.append('%.2d %.6d' % (seq, n))
        self.num_test = len(self.test_frames)
        
    def collect_train_frames(self):
        # adding info for both cameras
        self.train_frames = []
        for seq in self.train_seqs:
            img_dir = os.path.join(self.dataset_dir, '%.2d' % seq)
            N = len(glob(img_dir + '/*.npy'))
            for n in range(N):
                self.train_frames.append('%.2d %.6d' % (seq, n))
        self.num_train = len(self.train_frames)

    def is_valid_sample(self, frames, tgt_idx):
        N = len(frames)
        tgt_drive, _ = frames[tgt_idx].split(' ')
        half_offset = int((self.seq_length - 1)/2)
        min_src_idx = tgt_idx - half_offset
        max_src_idx = tgt_idx + half_offset
        if min_src_idx < 0 or max_src_idx >= N:
            return False
        min_src_drive, _ = frames[min_src_idx].split(' ')
        max_src_drive, _ = frames[max_src_idx].split(' ')
        if tgt_drive == min_src_drive and tgt_drive == max_src_drive:
            return True
        return False

    def load_image_sequence(self, frames, tgt_idx, seq_length):
        half_offset = int((seq_length - 1)/2)
        image_seq = []
        for o in range(-half_offset, half_offset+1):
            curr_idx = tgt_idx + o
            curr_drive, curr_frame_id = frames[curr_idx].split(' ')
            curr_img = self.load_image(curr_drive, curr_frame_id)
            # Normalize to values between -1 an 1
            curr_img = curr_img/MAX_DISPARITY
            curr_img = skimage.transform.resize(curr_img, (self.img_height, self.img_width))
            curr_img = curr_img*MAX_DISPARITY
            image_seq.append(curr_img)
        return image_seq

    def load_example(self, frames, tgt_idx, load_pose=False):
        image_seq = self.load_image_sequence(frames, tgt_idx, self.seq_length)
        tgt_drive, tgt_frame_id = frames[tgt_idx].split(' ')
        example = {}
        example['image_seq'] = image_seq
        example['folder_name'] = tgt_drive
        example['file_name'] = tgt_frame_id
        if load_pose:
            pass
        return example

    def get_train_example_with_idx(self, tgt_idx):
        if not self.is_valid_sample(self.train_frames, tgt_idx):
            return False
        example = self.load_example(self.train_frames, tgt_idx)
        return example

    def load_image(self, drive, frame_id):
        img_file = os.path.join(self.dataset_dir, '%s/%s.npy' % (drive, frame_id))
        img = np.load(img_file)
        return img
