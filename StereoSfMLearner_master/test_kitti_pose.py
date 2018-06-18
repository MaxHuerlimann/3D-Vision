from __future__ import division
import os
import cv2
import tensorflow as tf
import numpy as np
import skimage.transform
from glob import glob
from PSfMLearner import SfMLearner
from kitti_eval.pose_evaluation_utils import dump_pose_seq_TUM

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 1, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 384, "Image height")
flags.DEFINE_integer("img_width", 1280, "Image width")
flags.DEFINE_integer("seq_length", 5, "Sequence length for each example")
flags.DEFINE_integer("test_seq", 9, "Sequence id to test")
flags.DEFINE_string("dataset_dir", None, "Dataset directory")
flags.DEFINE_string("depths_dir", None, "Precalculated depth directory")
flags.DEFINE_string("output_dir", None, "Output directory")
#flags.DEFINE_string("ckpt_file", None, "checkpoint file")
flags.DEFINE_string("posenet_model", "./checkpoints/test_sep/", "Name of the trained posenet model")
flags.DEFINE_integer("max_disparity", 192, "Max disparity value from gcnet, used for depth normalization")
FLAGS = flags.FLAGS

def load_image_sequence(dataset_dir, 
                        depths_dir,
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
        curr_img = skimage.transform.resize(curr_img, (img_height, img_width))
        curr_img = np.expand_dims(curr_img, axis=0)
        if o == -half_offset:
            image_seq = curr_img
        else:
            image_seq = np.concatenate((image_seq, curr_img), axis=2)
    # same for depths
    for o in range(-half_offset, half_offset+1):
        curr_idx = tgt_idx + o
        curr_drive, curr_frame_id = frames[curr_idx].split(' ')
        img_file = os.path.join(
            depths_dir, '%s/%s.png' % (curr_drive, curr_frame_id))
#        curr_depth = np.load(img_file)
#        curr_depth = curr_depth/FLAGS.max_disparity
#        curr_depth = skimage.transform.resize(curr_depth, (img_height, img_width))
#        curr_depth = curr_depth*FLAGS.max_disparity
        curr_depth = cv2.imread(img_file, 0)
        curr_depth = np.expand_dims(curr_depth, axis=-1)
        curr_depth = skimage.transform.resize(curr_depth, (img_height, img_width))
        curr_depth = np.expand_dims(curr_depth, axis=0)
        if o == -half_offset:
            depth_seq = curr_depth
        else:
            depth_seq = np.concatenate((depth_seq, curr_depth), axis=2)
    return image_seq, depth_seq

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
    # Savers for posenet
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
#        restore_path=tf.train.latest_checkpoint(FLAGS.posenet_checkpoint_dir)
        saver.restore(sess, FLAGS.posenet_model)
#        saver.restore(sess, FLAGS.ckpt_file)
        for tgt_idx in range(N):
            if not is_valid_sample(test_frames, tgt_idx, FLAGS.seq_length):
                continue
            if tgt_idx % 100 == 0:
                print('Progress: %d/%d' % (tgt_idx, N))
            # TODO: currently assuming batch_size = 1
            image_seq, depth_seq = load_image_sequence(FLAGS.dataset_dir, 
                                            FLAGS.depths_dir,
                                            test_frames, 
                                            tgt_idx, 
                                            FLAGS.seq_length, 
                                            FLAGS.img_height, 
                                            FLAGS.img_width)
            pred = sfm.inference(image_seq, depth_seq, sess, mode='pose')
            pred_poses = pred['pose'][0]
            # Insert the target pose [0, 0, 0, 0, 0, 0] 
            pred_poses = np.insert(pred_poses, max_src_offset, np.zeros((1,6)), axis=0)
            curr_times = times[tgt_idx - max_src_offset:tgt_idx + max_src_offset + 1]
            out_file = FLAGS.output_dir + '%.6d.txt' % (tgt_idx - max_src_offset)
            dump_pose_seq_TUM(out_file, pred_poses, curr_times)

main()
