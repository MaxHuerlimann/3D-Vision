from __future__ import division
import tensorflow as tf
import pprint
import random
import numpy as np
from StereoSfMLearner import SfMLearner
import os
import argparse

# Reset argparse flags
tf.flags._global_parser = argparse.ArgumentParser()

flags = tf.app.flags
flags.DEFINE_string("dataset_dir", "G:\\KITTI_Od\\data_odometry_color\\formatted_dataset_small\\", "Dataset directory")
flags.DEFINE_string("checkpoint_dir", ".\\checkpoints\\test4\\", "Directory name to save the checkpoints")
flags.DEFINE_string("init_checkpoint_file", None, "Specific checkpoint file to initialize from")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam")
flags.DEFINE_float("smooth_weight", 0.5, "Weight for smoothness")
flags.DEFINE_float("explain_reg_weight", 0.0, "Weight for explanability regularization")
flags.DEFINE_integer("batch_size", 1, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 384, "Image height")
flags.DEFINE_integer("img_width", 1280, "Image width")
flags.DEFINE_integer("seq_length", 3, "Sequence length for each example")
flags.DEFINE_integer("max_steps", 1000, "Maximum number of training iterations")
flags.DEFINE_integer("summary_freq", 100, "Logging every log_freq iterations")
flags.DEFINE_integer("save_latest_freq", 500,
    "Save the latest model every save_latest_freq iterations (overwrites the previous latest model)")
flags.DEFINE_boolean("continue_train", False, "Continue training from previous checkpoint")

# Flags for gcnet parameters
flags.DEFINE_integer("max_disparity", 192, "The maximum disparity value")
flags.DEFINE_string("gcnet_model_dir", "./models/pretrained_gcnet/", "Pretrained GCNet directory")

FLAGS = flags.FLAGS

def main(_):
    seed = 8964
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    pp = pprint.PrettyPrinter()
    pp.pprint(flags.FLAGS.__flags)
    
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
        
    sfm = SfMLearner()
    sfm.train(FLAGS)

if __name__ == '__main__':
    tf.app.run()