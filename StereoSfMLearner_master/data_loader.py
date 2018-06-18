from __future__ import division
import os
import random
import cv2
import csv
import numpy as np
import tensorflow as tf

class DataLoader(object):
    def __init__(self, 
                 dataset_dir=None, 
                 depths_dir=None,
                 batch_size=None, 
                 img_height=None, 
                 img_width=None, 
                 num_source=None, 
                 num_scales=None):
        self.dataset_dir = dataset_dir
        self.depths_dir = depths_dir
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.num_source = num_source
        self.num_scales = num_scales
        self.seq_length = num_source + 1

    def make_intrinsics_matrix(self, fx, fy, cx, cy):
        # Assumes batch input
        batch_size = fx.get_shape().as_list()[0]
        zeros = tf.zeros_like(fx)
        r1 = tf.stack([fx, zeros, cx], axis=1)
        r2 = tf.stack([zeros, fy, cy], axis=1)
        r3 = tf.constant([0.,0.,1.], shape=[1, 3])
        r3 = tf.tile(r3, [batch_size, 1])
        intrinsics = tf.stack([r1, r2, r3], axis=1)
        return intrinsics

    def data_augmentation(self, im, intrinsics, out_h, out_w, seed, is_depth=False):
        # Random scaling
        def random_scaling(im, intrinsics, seed):
            batch_size, in_h, in_w, _ = im.get_shape().as_list()
            scaling = tf.random_uniform([2], 1, 1.15, seed=seed)
            x_scaling = scaling[0]
            y_scaling = scaling[1]
            out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
            out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)
            im = tf.image.resize_area(im, [out_h, out_w])
            fx = intrinsics[:,0,0] * x_scaling
            fy = intrinsics[:,1,1] * y_scaling
            cx = intrinsics[:,0,2] * x_scaling
            cy = intrinsics[:,1,2] * y_scaling
            intrinsics = self.make_intrinsics_matrix(fx, fy, cx, cy)
            return im, intrinsics

        # Random cropping
        def random_cropping(im, intrinsics, out_h, out_w, seed):
            # batch_size, in_h, in_w, _ = im.get_shape().as_list()
            batch_size, in_h, in_w, _ = tf.unstack(tf.shape(im))
            offset_y = tf.random_uniform([1], 0, in_h - out_h + 1, dtype=tf.int32, seed=seed)[0]
            offset_x = tf.random_uniform([1], 0, in_w - out_w + 1, dtype=tf.int32, seed=seed)[0]
            im = tf.image.crop_to_bounding_box(
                im, offset_y, offset_x, out_h, out_w)
            fx = intrinsics[:,0,0]
            fy = intrinsics[:,1,1]
            cx = intrinsics[:,0,2] - tf.cast(offset_x, dtype=tf.float32)
            cy = intrinsics[:,1,2] - tf.cast(offset_y, dtype=tf.float32)
            intrinsics = self.make_intrinsics_matrix(fx, fy, cx, cy)
            return im, intrinsics
        im, intrinsics = random_scaling(im, intrinsics, seed)
        im, intrinsics = random_cropping(im, intrinsics, out_h, out_w, seed)
        if not is_depth:
            im = tf.cast(im, dtype=tf.uint8)
        return im, intrinsics

    def format_file_list(self, data_root, depths_root, split, camera):
        with open(data_root + '/%s_%d.txt' % (split, camera), 'r') as f:
            frames = f.readlines()
        subfolders = [x.split(' ')[0] for x in frames]
        frame_ids = [x.split(' ')[1][:-1] for x in frames]
        image_file_list = [os.path.join(data_root, subfolders[i], 
            frame_ids[i] + '.jpg') for i in range(len(frames))]
        depths_file_list = [os.path.join(depths_root, subfolders[i][0:2], 
            frame_ids[i] + '.npy') for i in range(len(frames))]
        cam_file_list = [os.path.join(data_root, subfolders[i], 
            frame_ids[i] + '_cam.txt') for i in range(len(frames))]
        all_list = {}
        all_list['image_file_list'] = image_file_list
        all_list['cam_file_list'] = cam_file_list
        all_list['depths_file_list'] = depths_file_list
        return all_list

    def unpack_image_sequence(self, image_seq, img_height, img_width, num_source):
        # Assuming the center image is the target frame
        image_seq = tf.squeeze(image_seq)
        tgt_start_idx = int(img_width * (num_source//2))
        tgt_image = tf.slice(image_seq, 
                             [0, tgt_start_idx, 0], 
                             [-1, img_width, -1])
        num_channels = tgt_image.get_shape()[-1].value
        # Source frames before the target frame
        src_image_1 = tf.slice(image_seq, 
                               [0, 0, 0], 
                               [-1, int(img_width * (num_source//2)), -1])
        # Source frames after the target frame
        src_image_2 = tf.slice(image_seq, 
                               [0, int(tgt_start_idx + img_width), 0], 
                               [-1, int(img_width * (num_source//2)), -1])
        src_image_seq = tf.concat([src_image_1, src_image_2], axis=1)
        # Stack source frames along the color channels (i.e. [H, W, N*3])
        src_image_stack = tf.concat([tf.slice(src_image_seq, 
                                    [0, i*img_width, 0], 
                                    [-1, img_width, -1]) 
                                    for i in range(num_source)], axis=2)
        src_image_stack.set_shape([img_height, 
                                   img_width, 
                                   num_source * num_channels])
        tgt_image.set_shape([img_height, img_width, num_channels])
        
        return tgt_image, src_image_stack

    def batch_unpack_image_sequence(self, image_seq, img_height, img_width, num_source):
        # Assuming the center image is the target frame
        tgt_start_idx = int(img_width * (num_source//2))
        tgt_image = tf.slice(image_seq, 
                             [0, 0, tgt_start_idx, 0], 
                             [-1, -1, img_width, -1])
        # Source frames before the target frame
        src_image_1 = tf.slice(image_seq, 
                               [0, 0, 0, 0], 
                               [-1, -1, int(img_width * (num_source//2)), -1])
        # Source frames after the target frame
        src_image_2 = tf.slice(image_seq, 
                               [0, 0, int(tgt_start_idx + img_width), 0], 
                               [-1, -1, int(img_width * (num_source//2)), -1])
        src_image_seq = tf.concat([src_image_1, src_image_2], axis=2)
        # Stack source frames along the color channels (i.e. [B, H, W, N*3])
        src_image_stack = tf.concat([tf.slice(src_image_seq, 
                                    [0, 0, i*img_width, 0], 
                                    [-1, -1, img_width, -1]) 
                                    for i in range(num_source)], axis=3)
        return tgt_image, src_image_stack

    def get_multi_scale_intrinsics(self, intrinsics, num_scales):
        intrinsics_mscale = []
        # Scale the intrinsics accordingly for each scale
        for s in range(num_scales):
            fx = intrinsics[:,0,0]/(2 ** s)
            fy = intrinsics[:,1,1]/(2 ** s)
            cx = intrinsics[:,0,2]/(2 ** s)
            cy = intrinsics[:,1,2]/(2 ** s)
            intrinsics_mscale.append(
                self.make_intrinsics_matrix(fx, fy, cx, cy))
        intrinsics_mscale = tf.stack(intrinsics_mscale, axis=1)
        return intrinsics_mscale
            
    def load_batches(self, file_paths, disp_paths, cam_paths, step):
        num_files = len(file_paths)
        img_batch = []
        cam_batch = []
        depth_batch = []
        start_idx = step*self.batch_size
        for i in range(start_idx,start_idx+self.batch_size):
            # load image sequences
            img_seq = cv2.imread(file_paths[(i-1)%num_files], 1)
            img_seq = cv2.cvtColor(img_seq, cv2.COLOR_BGR2RGB)
            img_seq = np.expand_dims(img_seq, axis=0)
            img_batch.append(img_seq)
            # load raw camera intrinsics
            raw_cam_vec = self.load_raw_cam_vec(cam_paths[(i-1)%num_files])
            raw_cam_vec = np.expand_dims(raw_cam_vec, axis=0)
            cam_batch.append(raw_cam_vec)
            # load precalculated disparites and form into depths
            disp_seq = np.load(disp_paths[(i-1)%num_files])
            depth_seq = 1/disp_seq
            depth_seq = np.expand_dims(depth_seq, axis=-1)
            depth_seq = np.expand_dims(depth_seq, axis=0)
            depth_batch.append(depth_seq)
        # Concatenate all the lists into batches
        img_batch = np.concatenate(img_batch,0)
        cam_batch = np.concatenate(cam_batch,0)
        depth_batch = np.concatenate(depth_batch,0)
        return img_batch, depth_batch, cam_batch
    
    def load_raw_cam_vec(self, file_path):
        with open(file_path, 'r') as cam:
            reader = csv.reader(cam)
            cam_vec = list(reader)
            raw_cam_vec = np.reshape(np.array(cam_vec), (9,))
        return raw_cam_vec
    
    def shuffle_files(self, all_list):
        img_file_list = all_list['image_file_list']
        cam_file_list = all_list['cam_file_list']
        depths_file_list = all_list['depths_file_list']
        file_lists = list(zip(img_file_list, cam_file_list, depths_file_list))
        random.shuffle(file_lists)
        img_file_list_sh, cam_file_list_sh, depths_file_list_sh = zip(*file_lists)
        return img_file_list_sh, cam_file_list_sh, depths_file_list_sh

    def augment(self, image_seq, raw_cam_vec, depths, seed):
        # Unpack image sequence
        tgt_image, src_image_stack = \
                self.batch_unpack_image_sequence(
                        image_seq, self.img_height, self.img_width, self.num_source)

        # Unpack depth sequence
        tgt_image_depth, src_image_stack_depth = \
                self.batch_unpack_image_sequence(
                        depths, self.img_height, self.img_width, self.num_source)

        # Load camera intrinsics
        intrinsics = tf.reshape(raw_cam_vec, [self.batch_size,3, 3])

        # Augment depth        
        depth_all = tf.concat([tgt_image_depth, src_image_stack_depth], axis=3)
        depth_all, _ = self.data_augmentation(
            depth_all, intrinsics, self.img_height, self.img_width, seed, is_depth=True)
        tgt_image_depth = depth_all[:, :, :, :1]
        src_image_stack_depth = depth_all[:, :, :, 1:]
 
        # Data augmentation
        image_all = tf.concat([tgt_image, src_image_stack], axis=3)
        image_all, intrinsics = self.data_augmentation(
            image_all, intrinsics, self.img_height, self.img_width, seed)
        tgt_image = image_all[:, :, :, :3]
        src_image_stack = image_all[:, :, :, 3:]
        intrinsics = self.get_multi_scale_intrinsics(
            intrinsics, self.num_scales)
        
        # To define epoch size
        file_list = self.format_file_list(self.dataset_dir, self.depths_dir, 'train', 2)
        self.steps_per_epoch = int(
            len(file_list['image_file_list'])//self.batch_size)
        return tgt_image, src_image_stack, intrinsics, tgt_image_depth, src_image_stack_depth

    def scale_down(self, tensor, factor):
        h = tensor.get_shape()[1].value
        w = tensor.get_shape()[2].value
        out_h = tf.cast(h/factor, dtype=tf.int32)
        out_w = tf.cast(w/factor, dtype=tf.int32)
        tensor_scaled = tf.image.resize_area(tensor,[out_h,out_w])
        return tensor_scaled
    
    def process_img_seq(self, image_seq, raw_cam_vec, depths):
        # Seed to ensure all the images get augmented the same way
        seed = random.randint(1,2**31 - 1)

        # Augment data
        tgt_image, src_image_stack, intrinsics, tgt_image_depth, src_image_stack_depth = self.augment(image_seq, raw_cam_vec, depths, seed)

        # Scale depth for multiscale training, if used
        tgt_depth_augm_scaled_list = []
        tgt_depth_augm_scaled_list.append(tgt_image_depth)
        print(tgt_image_depth.get_shape())
        for i in range(3):
            tgt_depth_augm_scaled_list.append(self.scale_down(tgt_depth_augm_scaled_list[i], 2))

        return tgt_image, src_image_stack, intrinsics, tgt_image_depth, src_image_stack_depth, tgt_depth_augm_scaled_list
    
    # stack depths as fourth channel on RGB images, depths_all is a stack of all the full scale depth estimations
    def stack_rgbd(self, tgt_img, src_img_stack, depth_tgt_img, depth_src_img_all):
        # Stack target image depth on RGB image
        tgt_img_rgbd = tf.concat([tgt_img, depth_tgt_img], axis=3)
        
        # stack source images
        src_img_stack_rgbd = []
        for src in range(self.num_source):
            src_img = tf.slice(src_img_stack, 
                                 [0, 0, 0, src*3], 
                                 [-1, -1, -1, 3])
            depth_src_img = tf.slice(depth_src_img_all,[0,0,0,src],[-1,-1,-1,1])
            src_img_rgbd = tf.concat([src_img, depth_src_img], axis=3)
            src_img_stack_rgbd.append(src_img_rgbd)
        src_img_stack_rgbd = tf.concat(src_img_stack_rgbd, axis=3)

        return tgt_img_rgbd, src_img_stack_rgbd, tf.squeeze(depth_tgt_img)
