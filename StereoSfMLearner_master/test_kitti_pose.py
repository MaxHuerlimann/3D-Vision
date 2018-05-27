from __future__ import division
import os
import math
import scipy.misc
import cv2
import tensorflow as tf
import numpy as np
from glob import glob
from RGBDSfmLearner import SfMLearner
from kitti_eval.pose_evaluation_utils import dump_pose_seq_TUM

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 1, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 128, "Image height")
flags.DEFINE_integer("img_width", 416, "Image width")
flags.DEFINE_integer("seq_length", 3, "Sequence length for each example")
flags.DEFINE_integer("test_seq", 9, "Sequence id to test")
flags.DEFINE_string("dataset_dir", None, "Dataset directory")
flags.DEFINE_string("output_dir", None, "Output directory")
#flags.DEFINE_string("ckpt_file", None, "checkpoint file")
flags.DEFINE_string("posenet_checkpoint_dir", "./checkpoints/test_rgbd_new/", "Directory name to save the checkpoints")
flags.DEFINE_integer("max_disparity", 192, "The maximum disparity for gcnet")
flags.DEFINE_string("gcnet_model_dir", "./models/pretrained_gcnet/", "Pretrained GCNet directory")
flags.DEFINE_integer("num_source", None, "Number of source images")
flags.DEFINE_integer("num_scales", None, "Number of different disparity scales")
FLAGS = flags.FLAGS

def load_image_sequence(dataset_dir, 
                        frames, 
                        tgt_idx, 
                        seq_length, 
                        img_height, 
                        img_width):
    half_offset = int((seq_length - 1)/2)
    for o in range(-half_offset, half_offset+1):
        curr_idx = tgt_idx + o
        curr_drive, curr_frame_id = frames[curr_idx].split(' ')
        img_file = os.path.join(
            dataset_dir, 'sequences', '%s/image_2/%s.png' % (curr_drive, curr_frame_id))
        curr_img = cv2.imread(img_file)
        curr_img = scipy.misc.imresize(curr_img, (img_height, img_width))
        if o == -half_offset:
            image_seq_2 = curr_img
        else:
            image_seq_2 = np.hstack((image_seq_2, curr_img))
    # same for right image
    for o in range(-half_offset, half_offset+1):
        curr_idx = tgt_idx + o
        curr_drive, curr_frame_id = frames[curr_idx].split(' ')
        img_file = os.path.join(
            dataset_dir, 'sequences', '%s/image_3/%s.png' % (curr_drive, curr_frame_id))
        curr_img = cv2.imread(img_file)
        curr_img = scipy.misc.imresize(curr_img, (img_height, img_width))
        if o == -half_offset:
            image_seq_3 = curr_img
        else:
            image_seq_3 = np.hstack((image_seq_3, curr_img))
    return image_seq_2, image_seq_3

def is_valid_sample(frames, tgt_idx, seq_length):
    N = len(frames)
    tgt_drive, _ = frames[tgt_idx].split(' ')
    max_src_offset = int((seq_length - 1)/2)
    min_src_idx = tgt_idx - max_src_offset
    max_src_idx = tgt_idx + max_src_offset
    if min_src_idx < 0 or max_src_idx >= N:
        return False
    # TODO: unnecessary to check if the drives match
    min_src_drive, _ = frames[min_src_idx].split(' ')
    max_src_drive, _ = frames[max_src_idx].split(' ')
    if tgt_drive == min_src_drive and tgt_drive == max_src_drive:
        return True
    return False

def main():
    tf.reset_default_graph()
    sfm = SfMLearner()
    sfm.setup_inference(FLAGS,
                        'pose')
    # Savers for posenet and gcnet variables
    gcnet_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="stereo_network")
    gcnet_saver = tf.train.Saver(var_list = gcnet_var)
#    saver = tf.train.Saver([var for var in tf.trainable_variables()]) 
    posenet_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="pose_exp_net") + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="rgbd_emulation")
    saver = tf.train.Saver(var_list = posenet_var)

    if not os.path.isdir(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    seq_dir = os.path.join(FLAGS.dataset_dir, 'sequences', '%.2d' % FLAGS.test_seq)
    img_dir = os.path.join(seq_dir, 'image_2')
    N = len(glob(img_dir + '/*.png'))
    test_frames = ['%.2d %.6d' % (FLAGS.test_seq, n) for n in range(N)]
    with open(FLAGS.dataset_dir + 'sequences/%.2d/times.txt' % FLAGS.test_seq, 'r') as f:
        times = f.readlines()
    times = np.array([float(s[:-1]) for s in times])
    max_src_offset = (FLAGS.seq_length - 1)//2
    with tf.Session() as sess:
        restore_path=tf.train.latest_checkpoint(FLAGS.gcnet_model_dir)
        gcnet_saver.restore(sess, restore_path)
        restore_path=tf.train.latest_checkpoint(FLAGS.posenet_checkpoint_dir)
        saver.restore(sess, restore_path)
#        saver.restore(sess, FLAGS.ckpt_file)
        for tgt_idx in range(N):
            if not is_valid_sample(test_frames, tgt_idx, FLAGS.seq_length):
                continue
            if tgt_idx % 100 == 0:
                print('Progress: %d/%d' % (tgt_idx, N))
            # TODO: currently assuming batch_size = 1
            image_seq_2, image_seq_3 = load_image_sequence(FLAGS.dataset_dir, 
                                            test_frames, 
                                            tgt_idx, 
                                            FLAGS.seq_length, 
                                            FLAGS.img_height, 
                                            FLAGS.img_width)
            pred = sfm.inference(image_seq_2[None, :, :, :], image_seq_3[None, :, :, :], sess, FLAGS, mode='pose')
            pred_poses = pred['pose'][0]
            # Insert the target pose [0, 0, 0, 0, 0, 0] 
            pred_poses = np.insert(pred_poses, max_src_offset, np.zeros((1,6)), axis=0)
            curr_times = times[tgt_idx - max_src_offset:tgt_idx + max_src_offset + 1]
            out_file = FLAGS.output_dir + '%.6d.txt' % (tgt_idx - max_src_offset)
            dump_pose_seq_TUM(out_file, pred_poses, curr_times)

main()
