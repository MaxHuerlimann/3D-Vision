# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

from __future__ import absolute_import, division, print_function

# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import argparse
import re
import time
import tensorflow as tf
import cv2
from collections import namedtuple

from network import *
from GCDataloader import *
from pdb import set_trace as st

GC_parameters=namedtuple('parameters',
            'height, '
            'width, '
            'batch_size, '
            'num_threads, '
            'num_epochs, '
            'max_disparity, '
            'full_summary')


parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')

parser.add_argument('--mode',                      type=str,   help='train or test', default='test')
parser.add_argument('--model_name',                type=str,   help='model name', default='model_test')
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti, or cityscapes', default='kitti')
parser.add_argument('--data_path',                 type=str,   help='path to the data', default='.\\testingdata\\')
parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', default='.\\test_files.txt')
parser.add_argument('--input_height',              type=int,   help='input height', default=256)
parser.add_argument('--input_width',               type=int,   help='input width', default=512)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=1)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=100)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--max_disparity',             type=int,   help='maximum disparity', default=192)
parser.add_argument('--num_gpus',                  type=int,   help='number of GPUs to use for training', default=1)
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=8)
parser.add_argument('--output_directory',          type=str,   help='output directory for test disparities, if empty outputs to checkpoint folder', default='.\\test_result\\')
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='.\\log\\')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', default='')
parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
parser.add_argument('--full_summary',                          help='if set, will keep more data for each summary. Warning: the file can become very large', action='store_true')

args = parser.parse_args()

def count_text_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    lines = [l.rstrip() for l in lines]
    return len(lines),lines

def computeSoftArgMin(logits,h,w,d):
    softmax = tf.nn.softmax(logits)
    disp = tf.range(1, d+1, 1)
    disp = tf.cast(disp, tf.float32)
    disp_mat = []
    for i in range(w*h):
        disp_mat.append(disp)
    disp_mat.append(disp)
    disp_mat = tf.reshape(tf.stack(disp_mat), [h,w,d])
    result = tf.multiply(softmax, disp_mat)
    result = tf.reduce_sum(result, 2)
    return result


def disp_loss(left_logits, left_labels):
    left_loss_ = tf.abs(tf.subtract(left_logits, left_labels))
    left_mask = tf.cast(left_labels>0, dtype=tf.bool)
    left_loss_ = tf.where(left_mask, left_loss_, tf.zeros_like(left_loss_))
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    left_loss_sum = tf.reduce_sum(left_loss_)
    left_mask = tf.cast(left_mask, tf.float32)
    left_loss_mean = tf.div(left_loss_sum, tf.reduce_sum(left_mask))
    loss_final=tf.add_n([left_loss_mean]+regularization_losses)
    return loss_final

    
def train(params):
    """Training loop."""
    print("==========train========")
    with tf.Graph().as_default(), tf.device('/cpu:0'):

        global_step = tf.Variable(0, trainable=False)

        # OPTIMIZER
        num_training_samples,_ = count_text_lines(args.filenames_file)

        steps_per_epoch = np.ceil(num_training_samples / params.batch_size).astype(np.int32)
        num_total_steps = params.num_epochs * steps_per_epoch
        start_learning_rate = args.learning_rate

        boundaries = [np.int32((3/5) * num_total_steps), np.int32((4/5) * num_total_steps)]
        values = [args.learning_rate, args.learning_rate / 2, args.learning_rate / 4]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, epsilon=1e-8)

        print("total number of samples: {}".format(num_training_samples))
        print("total number of steps: {}".format(num_total_steps))

        dataloader = GCDataloader(args.data_path, args.filenames_file, params, args.dataset, args.mode)

        left_img  = dataloader.left_image_batch
        right_img = dataloader.right_image_batch
        left_labels= dataloader.label_batch
        left_img = left_img * 255.0
        right_img = right_img * 255.0
        sum_im=tf.image.convert_image_dtype(left_img*0.5+0.5,  tf.uint8)
        sum_label=tf.image.convert_image_dtype(left_labels,  tf.uint8)
        tf.summary.image('left_image',sum_im)
        tf.summary.image('label_image',sum_label)
        #right_labels=dataloader.right_labels
        #st()#break point

        # split for each gpu
        left_splits  = tf.split(left_img,  args.num_gpus, 0)
        right_splits = tf.split(right_img, args.num_gpus, 0)
        left_labels = tf.split(left_labels, args.num_gpus, 0)
        reuse_variables = None
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(args.num_gpus):
                with tf.device('/gpu:2'):
                    disp_pre = stereo_model((left_splits[i]/255.0-0.5)/0.5, (right_splits[i]/255.0-0.5)/0.5, params, is_training=True,
                                            rv=reuse_variables)
                    sum_predisp = tf.expand_dims(disp_pre, 2)
                    disp_pre = tf.image.resize_images(sum_predisp * 2, [args.input_height, args.input_width],
                                                      tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                    disp_pre = tf.squeeze(disp_pre)
                    print('disp__size' + str(disp_pre.get_shape()))
                    tf.summary.image('gt_disparity', tf.image.convert_image_dtype(left_labels[i], tf.uint8))
                    disp_labels = tf.squeeze(left_labels[i])
                    loss_disp = disp_loss(disp_pre, disp_labels * 65535.0 / 256.0)
                    total_loss = loss_disp
                    reuse_variables = True
                    grads = optimizer.compute_gradients(total_loss)

                    tf.summary.image('predict_disparity',
                                     tf.expand_dims(tf.image.convert_image_dtype(sum_predisp/255.0, tf.uint8), 0))

        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
        batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
        batchnorm_updates_op = tf.group(*batchnorm_updates)

        # train op contains optimizer, batchnorm and averaged loss
        train = tf.group(apply_gradient_op, batchnorm_updates_op)
        tf.summary.scalar('total_loss', total_loss)
        summary_op = tf.summary.merge_all()

        # SESSION
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth=True
        with tf.Session(config=config) as sess:
            # SAVER
            summary_writer = tf.summary.FileWriter(args.log_directory + '/' + "SFview", sess.graph)
            train_saver = tf.train.Saver(tf.global_variables())
            # COUNT PARAMS
            total_num_parameters = 0
            for variable in tf.trainable_variables():
                total_num_parameters += np.array(variable.get_shape().as_list()).prod()
            print("number of trainable parameters: {}".format(total_num_parameters))

            # INIT
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            coordinator = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

            # LOAD CHECKPOINT IF SET
            if args.checkpoint_path != '':
                train_saver.restore(sess, args.checkpoint_path.split(".")[0])

                if args.retrain:
                      sess.run(global_step.assign(0))
            # GO!
            start_step = global_step.eval(session=sess)
            start_time = time.time()
            for step in range(start_step, num_total_steps):
                before_op_time = time.time()
                _, loss_value= sess.run([train, total_loss])
                duration = time.time() - before_op_time
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, global_step=step)

                if step and step % 100 == 0:
                    examples_per_sec = params.batch_size / duration
                    time_sofar = (time.time() - start_time) / 3600
                    training_time_left = (num_total_steps / step - 1.0) * time_sofar
                    print_string = 'batch {:>6} | examples/s: {:4.2f} | loss: {:.5f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                    print(print_string.format(step, examples_per_sec, loss_value, time_sofar, training_time_left))

                if step and step % 10000 == 0:
                    train_saver.save(sess, args.log_directory + '/' + args.model_name + '/model', global_step=step)

            train_saver.save(sess, args.log_directory + '/' + args.model_name + '/model', global_step=num_total_steps)

def test(params):
    """Test function."""
    tf.reset_default_graph()
    print("=====================test===================")
    h=384
    w=1280
    left_img=tf.placeholder(tf.float32,shape=(1,h,w,3))
    right_img=tf.placeholder(tf.float32,shape=(1,h,w,3))

    disp_pre = stereo_model(left_img, right_img, params, is_training=True)
    disp_pre = tf.expand_dims(disp_pre, 2)
    disp_pre = tf.image.resize_images(disp_pre, [h, w], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    disp_pre = tf.squeeze(disp_pre)

    print('disp_pre ready')

    #SESSION
    config=tf.ConfigProto(allow_soft_placement=True)
    sess=tf.Session(config=config)

    #SAVER
    train_saver=tf.train.Saver()

    #INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator=tf.train.Coordinator()

    #RESTORE
    print('Starting to restore')
    if args.checkpoint_path=='':
        restore_path=tf.train.latest_checkpoint(args.log_directory+'/'+args.model_name)
    else:
        restore_path=args.checkpoint_path.split(".")[0]
    train_saver.restore(sess,restore_path)

    num_test_samples,filenames=count_text_lines(args.filenames_file)

    print('now test {} files'.format(num_test_samples))

    output_directory=args.output_directory

    for filename in filenames:
        file_ns = filename.split()
        splits = file_ns[0].split('/')
        img_id=splits[-1][:-4]
        print(img_id)
        imgL=cv2.imread(args.data_path+file_ns[0])
        imgR=cv2.imread(args.data_path+file_ns[1])
        shape=imgL.shape
        top_pad=h-shape[0]
        right_pad=w-shape[1]
        imgL=np.lib.pad(imgL,((top_pad,0),(0,right_pad),(0,0)),mode='constant',constant_values=0)
        imgR=np.lib.pad(imgR,((top_pad,0),(0,right_pad),(0,0)),mode='constant',constant_values=0)
        imgL = cv2.resize(imgL, (w, h), interpolation=cv2.INTER_NEAREST)
        imgR = cv2.resize(imgR, (w, h), interpolation=cv2.INTER_NEAREST)
        imgL=np.expand_dims(imgL,axis=0)
        imgR=np.expand_dims(imgR,axis=0)
        start_time=time.time()
        print('session started')
        disp = sess.run(disp_pre,
                        feed_dict={left_img: (imgL / 255.0 - 0.5) / 0.5, right_img: (imgR / 255.0 - 0.5) / 0.5})
        print('time = %.2f' %(time.time()-start_time))
        disp_=disp[top_pad:,:-right_pad]
        disparities=np.uint16(np.round(disp_*256.0))
        cv2.imwrite(output_directory+img_id+'.png',disparities)
    print('done.')


def main(_):
    params = GC_parameters(
        height=args.input_height,
        width=args.input_width,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        num_epochs=args.num_epochs,
        max_disparity=args.max_disparity,
        full_summary=args.full_summary)

    if args.mode == 'train':
        train(params)
    elif args.mode == 'test':
        test(params)

if __name__ == '__main__':
    tf.app.run()

