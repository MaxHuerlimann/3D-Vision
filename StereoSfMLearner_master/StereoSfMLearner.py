from __future__ import division
import os
import time
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from data_loader import DataLoader
from posenet import pose_exp_net
from gcnet import disp_net
from utils import *


class SfMLearner(object):
    def __init__(self):
        pass
    
    def build_train_graph(self, image_seq_2, image_seq_3, gcnet_img_2, gcnet_img_3, raw_cam_vec_2, raw_cam_vec_3, pred_depths_rgbd, which_img):
        print('Train graph is getting built')
        opt = self.opt
        loader = DataLoader(opt.dataset_dir,
                            opt.batch_size,
                            opt.img_height,
                            opt.img_width,
                            opt.num_source,
                            opt.num_scales)

        with tf.name_scope("data_unpacking"):
            print('Choose gcnet image')
#            image_seq_2 = self.preprocess_image(image_seq_2)
#            image_seq_3 = self.preprocess_image(image_seq_3)
#            gcnet_img_2 = loader.unpack_image_sequence_gcnet(image_seq_2, opt.img_height, opt.img_width, opt.num_source, which_img)
#            gcnet_img_3 = loader.unpack_image_sequence_gcnet(image_seq_3, opt.img_height, opt.img_width, opt.num_source, which_img)
#            image_seq_2 = self.deprocess_image(image_seq_2)
#            image_seq_3 = self.deprocess_image(image_seq_3)
            gcnet_img_2 =self.preprocess_image(gcnet_img_2)
            gcnet_img_3 =self.preprocess_image(gcnet_img_3)


        # Depth prediction with gcnet 
        with tf.name_scope("depth_prediction"):
            print('Depth prediction')
            pred_disp, depth_net_endpoints = disp_net(gcnet_img_2 ,gcnet_img_3, opt.max_disparity)
            pred_depth_gcnet = 1./pred_disp

        with tf.name_scope("data_augmentation"):
            print('Data is being augmented')
            tgt_image_augmented, src_image_stack_augmented, intrinsic, pred_depths_augmented, pred_depth = loader.process_img_seq(image_seq_2, raw_cam_vec_2, pred_depths_rgbd)
            # Preprocess left images
            tgt_image_2 = self.preprocess_image(tgt_image_augmented)
            src_image_stack_2 = self.preprocess_image(src_image_stack_augmented)
            intrinsics_2 = intrinsic

        with tf.name_scope("rgbd_emulation"):
            tgt_image_2, src_image_stack_2, depth_tgt_image = loader.stack_rgbd(tgt_image_2, src_image_stack_2, pred_depths_augmented)
            
        with tf.name_scope("pose_and_explainability_prediction"):
            print('Pose prediction started')
            pred_poses, pred_exp_logits, pose_exp_net_endpoints = \
                pose_exp_net(tgt_image_2,
                             src_image_stack_2, 
                             do_exp=(opt.explain_reg_weight > 0),
                             is_training=True)
            print('Pose prediction finished')
            

        with tf.name_scope("compute_loss"):
            print('Loss computation started')
            # Extract RGB channels  from target RGBD image
            tgt_image_2 = tf.slice(tgt_image_2,[0,0,0,0],[-1,-1,-1,3])
            # Split depth maps from source RGBD images
            rgb_src_img_stack = []
            depth_src_img_stack = []
            for img in range(opt.num_source):
                rgbd_src_img = tf.slice(src_image_stack_2,[0,0,0,img*4],[-1,-1,-1,4])
                rgb_src_img = tf.slice(rgbd_src_img,[0,0,0,0],[-1,-1,-1,3])
                depth_src_img = tf.slice(rgbd_src_img,[0,0,0,3],[-1,-1,-1,1])
                rgb_src_img_stack.append(rgb_src_img)
                depth_src_img_stack.append(depth_src_img)
            src_image_stack_2 = tf.concat(rgb_src_img_stack, axis=3)
            src_image_stack_depth = tf.concat(depth_src_img_stack, axis=3)
            
            # Loss computation
            pixel_loss = 0
            exp_loss = 0
            smooth_loss = 0
            tgt_image_all = []
#            tgt_image_all_3 = []
            src_image_stack_all = []
            proj_image_stack_all = []
            proj_error_stack_all = []
            exp_mask_stack_all = []
            for s in range(opt.num_scales):
                if opt.explain_reg_weight > 0:
                    # Construct a reference explainability mask (i.e. all 
                    # pixels are explainable)
                    ref_exp_mask = self.get_reference_explain_mask(s)
                # Scale the source and target images for computing loss at the 
                # according scale.
                curr_tgt_image = tf.image.resize_area(tgt_image_2, 
                    [int(opt.img_height/(2**s)), int(opt.img_width/(2**s))])
#                curr_tgt_image_3 = tf.image.resize_area(tgt_image_3, 
#                    [int(opt.img_height/(2**s)), int(opt.img_width/(2**s))])                
                curr_src_image_stack = tf.image.resize_area(src_image_stack_2, 
                    [int(opt.img_height/(2**s)), int(opt.img_width/(2**s))])

                if opt.smooth_weight > 0:
                    smooth_loss += opt.smooth_weight/(2**s) * \
                        self.compute_smooth_loss(pred_disp[0][s])

                for i in range(opt.num_source):
                    # Inverse warp the source image to the target image frame
                    curr_proj_image = projective_inverse_warp(
                        curr_src_image_stack[:,:,:,3*i:3*(i+1)], 
                        tf.expand_dims(depth_tgt_image, 0), 
                        pred_poses[:,i,:], 
                        intrinsics_2[:,s,:,:])
                    curr_proj_error = tf.abs(curr_proj_image - curr_tgt_image)
                    # Cross-entropy loss as regularization for the 
                    # explainability prediction
                    if opt.explain_reg_weight > 0:
                        curr_exp_logits = tf.slice(pred_exp_logits[s], 
                                                   [0, 0, 0, i*2], 
                                                   [-1, -1, -1, 2])
                        exp_loss += opt.explain_reg_weight * \
                            self.compute_exp_reg_loss(curr_exp_logits,
                                                      ref_exp_mask)
                        curr_exp = tf.nn.softmax(curr_exp_logits)
                    # Photo-consistency loss weighted by explainability
                    if opt.explain_reg_weight > 0:
                        pixel_loss += tf.reduce_mean(curr_proj_error * \
                            tf.expand_dims(curr_exp[:,:,:,1], -1))
                    else:
                        pixel_loss += tf.reduce_mean(curr_proj_error) 
                    # Prepare images for tensorboard summaries
                    if i == 0:
                        proj_image_stack = curr_proj_image
                        proj_error_stack = curr_proj_error
                        if opt.explain_reg_weight > 0:
                            exp_mask_stack = tf.expand_dims(curr_exp[:,:,:,1], -1)
                    else:
                        proj_image_stack = tf.concat([proj_image_stack, 
                                                      curr_proj_image], axis=3)
                        proj_error_stack = tf.concat([proj_error_stack, 
                                                      curr_proj_error], axis=3)
                        if opt.explain_reg_weight > 0:
                            exp_mask_stack = tf.concat([exp_mask_stack, 
                                tf.expand_dims(curr_exp[:,:,:,1], -1)], axis=3)
                tgt_image_all.append(curr_tgt_image)
#                tgt_image_all_3.append(curr_tgt_image_3)
                src_image_stack_all.append(curr_src_image_stack)
                proj_image_stack_all.append(proj_image_stack)
                proj_error_stack_all.append(proj_error_stack)
                if opt.explain_reg_weight > 0:
                    exp_mask_stack_all.append(exp_mask_stack)
            total_loss = pixel_loss + smooth_loss + exp_loss

        with tf.name_scope("train_op"):
            # train_vars = [var for var in tf.trainable_variables()]
            optim = tf.train.AdamOptimizer(opt.learning_rate, opt.beta1)
            # self.grads_and_vars = optim.compute_gradients(total_loss, 
            #                                               var_list=train_vars)
            # self.train_op = optim.apply_gradients(self.grads_and_vars)
            posenet_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="pose_exp_net")
            self.train_op = slim.learning.create_train_op(total_loss, optim, variables_to_train=posenet_var)
            self.global_step = tf.Variable(0, 
                                           name='global_step', 
                                           trainable=False)
            self.incr_global_step = tf.assign(self.global_step, 
                                              self.global_step+1)

        # Collect tensors that are useful later (e.g. tf summary)
        self.gcnet_img_2 = gcnet_img_2
        self.gcnet_img_3 = gcnet_img_3
        self.pred_depth_gcnet = pred_depth_gcnet
        self.pred_depth = pred_depth
        self.pred_poses = pred_poses
        self.steps_per_epoch = loader.steps_per_epoch
        self.total_loss = total_loss
        self.pixel_loss = pixel_loss
        self.exp_loss = exp_loss
        self.smooth_loss = smooth_loss
        self.tgt_image_all = tgt_image_all
#        self.tgt_image_all_3 = tgt_image_all_3
        self.src_image_stack_all = src_image_stack_all
        self.proj_image_stack_all = proj_image_stack_all
        self.proj_error_stack_all = proj_error_stack_all
        self.exp_mask_stack_all = exp_mask_stack_all
        self.src_image_stack_depth = src_image_stack_depth

    def get_reference_explain_mask(self, downscaling):
        opt = self.opt
        tmp = np.array([0,1])
        ref_exp_mask = np.tile(tmp, 
                               (opt.batch_size, 
                                int(opt.img_height/(2**downscaling)), 
                                int(opt.img_width/(2**downscaling)), 
                                1))
        ref_exp_mask = tf.constant(ref_exp_mask, dtype=tf.float32)
        return ref_exp_mask

    def compute_exp_reg_loss(self, pred, ref):
        l = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.reshape(ref, [-1, 2]),
            logits=tf.reshape(pred, [-1, 2]))
        return tf.reduce_mean(l)

    def compute_smooth_loss(self, pred_disp):
        def gradient(pred):
            D_dy = pred[ 1: , :] - pred[ :-1, :]
            D_dx = pred[ : , 1:] - pred[ : , :-1]
            return D_dx, D_dy
        dx, dy = gradient(pred_disp)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        return tf.reduce_mean(tf.abs(dx2)) + \
               tf.reduce_mean(tf.abs(dxdy)) + \
               tf.reduce_mean(tf.abs(dydx)) + \
               tf.reduce_mean(tf.abs(dy2))

    def collect_summaries(self):
        opt = self.opt
        tf.summary.scalar("total_loss", self.total_loss)
        tf.summary.scalar("pixel_loss", self.pixel_loss)
        tf.summary.scalar("smooth_loss", self.smooth_loss)
        tf.summary.scalar("exp_loss", self.exp_loss)
#        tf.summary.image('gcnet_tgt_image_2', self.deprocess_image(self.gcnet_img_2))
#        tf.summary.image('gcnet_tgt_image_3', self.deprocess_image(self.gcnet_img_3))
        for i in range(opt.num_source):
            tf.summary.image('source_depth_image_%d' % i,
                    1./self.src_image_stack_depth[:, :, :, i:(i+1)])
        for s in range(opt.num_scales):
            tf.summary.histogram("scale%d_depth" % s, self.pred_depth[s])
            pred_depth_sum = tf.expand_dims(self.pred_depth[s],0)
            pred_depth_sum = tf.expand_dims(pred_depth_sum,3)
            tf.summary.image('scale%d_disparity_image' % s, 1./pred_depth_sum)
            tf.summary.image('scale%d_target_image' % s, \
                             self.deprocess_image(self.tgt_image_all[s]))
#            tf.summary.image('scale%d_target_image_3' % s, \
#                             self.deprocess_image(self.tgt_image_all_3[s]))
            for i in range(opt.num_source):
                if opt.explain_reg_weight > 0:
                    tf.summary.image(
                        'scale%d_exp_mask_%d' % (s, i), 
                        tf.expand_dims(self.exp_mask_stack_all[s][:,:,:,i], -1))
                tf.summary.image(
                    'scale%d_source_image_%d' % (s, i), 
                    self.deprocess_image(self.src_image_stack_all[s][:, :, :, i*3:(i+1)*3]))
                tf.summary.image('scale%d_projected_image_%d' % (s, i), 
                    self.deprocess_image(self.proj_image_stack_all[s][:, :, :, i*3:(i+1)*3]))
                tf.summary.image('scale%d_proj_error_%d' % (s, i),
                    self.deprocess_image(tf.clip_by_value(self.proj_error_stack_all[s][:,:,:,i*3:(i+1)*3] - 1, -1, 1)))
        tf.summary.histogram("tx", self.pred_poses[:,:,0])
        tf.summary.histogram("ty", self.pred_poses[:,:,1])
        tf.summary.histogram("tz", self.pred_poses[:,:,2])
        tf.summary.histogram("rx", self.pred_poses[:,:,3])
        tf.summary.histogram("ry", self.pred_poses[:,:,4])
        tf.summary.histogram("rz", self.pred_poses[:,:,5])
        # for var in tf.trainable_variables():
        #     tf.summary.histogram(var.op.name + "/values", var)
        # for grad, var in self.grads_and_vars:
        #     tf.summary.histogram(var.op.name + "/gradients", grad)
        
        
        

    def train(self, opt):
        # Reset graph so can be called several times (debugging)
        tf.reset_default_graph()
        opt.num_source = opt.seq_length - 1
        # TODO: currently fixed to 1, as GCNet doesn't include multiscales
        opt.num_scales = 4
        self.opt = opt

        # Initialize placeholders
        image_seq_2 = tf.placeholder(tf.float32,shape=(1,opt.img_height,opt.seq_length*opt.img_width,3))
        image_seq_3 = tf.placeholder(tf.float32,shape=(1,opt.img_height,opt.seq_length*opt.img_width,3))
        gcnet_img_2 = tf.placeholder(tf.uint8, shape=(1, opt.img_height, opt.img_width,3))
        gcnet_img_3 = tf.placeholder(tf.uint8, shape=(1, opt.img_height, opt.img_width,3))
        cam_vec_2 = tf.placeholder(tf.float32, shape=(9))
        cam_vec_3 = tf.placeholder(tf.float32, shape=(9))
        pred_depths_input = tf.placeholder(tf.float32,shape=(opt.seq_length, opt.img_height,opt.img_width))
        which_img = tf.placeholder(tf.int32,shape=())
        
        # Build the graph
        self.build_train_graph(image_seq_2, image_seq_3, gcnet_img_2, gcnet_img_3, cam_vec_2, cam_vec_3, pred_depths_input, which_img)
        self.collect_summaries()      
        print('graph built')
        
        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                            for v in tf.trainable_variables()])
        self.saver = tf.train.Saver([var for var in tf.model_variables()] + \
                                    [self.global_step],
                                     max_to_keep=10)
        gcnet_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="stereo_network")
        self.gcnet_saver = tf.train.Saver(var_list = gcnet_var)
        
        sv = tf.train.Supervisor(logdir=opt.checkpoint_dir, 
                                 save_summaries_secs=0, 
                                 saver=None)
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with sv.managed_session(config=config) as sess:
            #INIT
            restore_path=tf.train.latest_checkpoint(opt.gcnet_model_dir)
            self.gcnet_saver.restore(sess, restore_path)
#            print('Trainable variables: ')
#            for var in tf.trainable_variables():
#                print(var.name)
            print("parameter_count =", sess.run(parameter_count))
            if opt.continue_train:
                if opt.init_checkpoint_file is None:
                    checkpoint = tf.train.latest_checkpoint(opt.checkpoint_dir)
                else:
                    checkpoint = opt.init_checkpoint_file
                print("Resume training from previous checkpoint: %s" % checkpoint)
#                self.saver.restore(sess, checkpoint)
            print('Checkpoints checked')
            
            # Load file lists
            gc_dataloader = DataLoader(opt.dataset_dir,
                            opt.batch_size,
                            opt.img_height,
                            opt.img_width,
                            opt.num_source,
                            opt.num_scales)
            all_list_2 = gc_dataloader.format_file_list(opt.dataset_dir, 'train', 2)
            all_list_3 = gc_dataloader.format_file_list(opt.dataset_dir, 'train', 3)
             
            # Shuffle the files
            file_paths_2, file_paths_3, cam_paths_2, cam_paths_3 = gc_dataloader.shuffle_files(all_list_2, all_list_3)
            num_files = len(file_paths_2)
            
            start_time = time.time()
            for step in range(1, opt.max_steps):
                # Loading the image sequence
                image_sequence_left, image_sequence_right = gc_dataloader.load_gcnet_img(file_paths_2[(step-1)%num_files], file_paths_3[(step-1)%num_files])
                raw_cam_vec_2 = gc_dataloader.load_raw_cam_vec(cam_paths_2[(step-1)%num_files])
                raw_cam_vec_3 = gc_dataloader.load_raw_cam_vec(cam_paths_3[(step-1)%num_files])
                image_sequence_left = np.expand_dims(image_sequence_left, axis=0)
                image_sequence_right = np.expand_dims(image_sequence_right, axis=0)
                
                # Running gcnet for all the pictures in the sequence
                fetches_gcnet = { "pred_depth_gcnet": self.pred_depth_gcnet}
                # Dummy array for feed_dict
                dummy_depths = np.zeros((opt.seq_length, opt.img_height, opt.img_width), dtype=np.float32)
                pred_depths_feed = []
                for src in range(opt.seq_length):
                    gcnet_img_left = gc_dataloader.unpack_image_sequence_gcnet(image_sequence_left, opt.img_height, opt.img_width, opt.num_source, src)
                    gcnet_img_right = gc_dataloader.unpack_image_sequence_gcnet(image_sequence_right, opt.img_height, opt.img_width, opt.num_source, src)
                    feed_dict_gcnet = {
                            image_seq_2: image_sequence_left,
                            image_seq_3: image_sequence_right,
                            gcnet_img_2: gcnet_img_left,
                            gcnet_img_3: gcnet_img_right,
                            cam_vec_2: raw_cam_vec_2,
                            cam_vec_3: raw_cam_vec_3,
                            pred_depths_input: dummy_depths,
                            which_img: src}
                    results_gcnet = sess.run(fetches_gcnet, feed_dict=feed_dict_gcnet)
#                    print('results_gcnet shape: ' + str(results_gcnet["pred_depth_gcnet"].shape))
                    pred_depths_feed.append(results_gcnet["pred_depth_gcnet"])
                pred_depths_feed = np.stack(pred_depths_feed,axis=0)
#                print('pred_depths_feed shape: ' + str(pred_depths_feed.shape))
                
                # Running posenet
                fetches = {
                    "train": self.train_op,
                    "global_step": self.global_step,
                    "incr_global_step": self.incr_global_step
                }
                
                if step % opt.summary_freq == 0:
                    fetches["loss"] = self.total_loss
                    fetches["summary"] = sv.summary_op
                feed_dict_posenet = {
                        image_seq_2: image_sequence_left,
                        image_seq_3: image_sequence_right,
                        gcnet_img_2: gcnet_img_left,
                        gcnet_img_3: gcnet_img_right,
                        cam_vec_2: raw_cam_vec_2,
                        cam_vec_3: raw_cam_vec_3,
                        pred_depths_input: pred_depths_feed,
                        which_img: opt.num_source/2}
                results = sess.run(fetches, feed_dict=feed_dict_posenet)
                gs = results["global_step"]

                if step % opt.summary_freq == 0:
                    sv.summary_writer.add_summary(results["summary"], gs)
                    train_epoch = math.ceil(gs / self.steps_per_epoch)
                    train_step = gs - (train_epoch - 1) * self.steps_per_epoch
                    print("Epoch: [ %2d] [%5d/%5d] time: %4.4f/it loss: %.3f" \
                            % (train_epoch, train_step, self.steps_per_epoch, \
                                (time.time() - start_time)/opt.summary_freq, 
                                results["loss"]))
                    start_time = time.time()

                if step % opt.save_latest_freq == 0:
                    self.save(sess, opt.checkpoint_dir, 'latest')

                if step % self.steps_per_epoch == 0:
                    self.save(sess, opt.checkpoint_dir, gs)

                if step % opt.save_model_freq == 0:
                    self.save(sess, opt.checkpoint_dir, gs)

    def build_depth_test_graph(self):
        input_uint8 = tf.placeholder(tf.uint8, [self.batch_size, 
                    self.img_height, self.img_width, 3], name='raw_input')
        input_mc = self.preprocess_image(input_uint8)
        with tf.name_scope("depth_prediction"):
            pred_disp, depth_net_endpoints = disp_net(
                input_mc, is_training=False)
            pred_depth = [1./disp for disp in pred_disp]
        pred_depth = pred_depth[0]
        self.inputs = input_uint8
        self.pred_depth = pred_depth
        self.depth_epts = depth_net_endpoints

    def build_pose_test_graph(self):
        input_uint8 = tf.placeholder(tf.uint8, [self.batch_size, 
            self.img_height, self.img_width * self.seq_length, 3], 
            name='raw_input')
        input_mc = self.preprocess_image(input_uint8)
        loader = DataLoader()
        tgt_image, src_image_stack = \
            loader.batch_unpack_image_sequence(
                input_mc, self.img_height, self.img_width, self.num_source)
        with tf.name_scope("pose_prediction"):
            pred_poses, _, _ = pose_exp_net(
                tgt_image, src_image_stack, do_exp=False, is_training=False)
            self.inputs = input_uint8
            self.pred_poses = pred_poses

    def preprocess_image(self, image):
        # Assuming input image is uint8
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image * 2. -1.

    def deprocess_image(self, image):
        # Assuming input image is float32
        image = (image + 1.)/2.
        return tf.image.convert_image_dtype(image, dtype=tf.uint8)

    def setup_inference(self, 
                        img_height,
                        img_width,
                        mode,
                        seq_length=3,
                        batch_size=1):
        self.img_height = img_height
        self.img_width = img_width
        self.mode = mode
        self.batch_size = batch_size
        if self.mode == 'depth':
            self.build_depth_test_graph()
        if self.mode == 'pose':
            self.seq_length = seq_length
            self.num_source = seq_length - 1
            self.build_pose_test_graph()

    def inference(self, inputs, sess, mode='depth'):
        fetches = {}
        if mode == 'depth':
            fetches['depth'] = self.pred_depth
        if mode == 'pose':
            fetches['pose'] = self.pred_poses
        results = sess.run(fetches, feed_dict={self.inputs:inputs})
        return results

    def save(self, sess, checkpoint_dir, step):
        model_name = 'model'
        print(" [*] Saving checkpoint to %s..." % checkpoint_dir)
        if step == 'latest':
            self.saver.save(sess, 
                            os.path.join(checkpoint_dir, model_name + '.latest'))
        else:
            self.saver.save(sess, 
                            os.path.join(checkpoint_dir, model_name),
                            global_step=step)
        
