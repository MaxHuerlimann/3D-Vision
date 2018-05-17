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
                 batch_size=None, 
                 img_height=None, 
                 img_width=None, 
                 num_source=None, 
                 num_scales=None):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.num_source = num_source
        self.num_scales = num_scales

    def load_and_augment(self, file_list, seed):
        random.seed(seed)
        random.shuffle(file_list['image_file_list'])
        image_paths_queue = tf.train.string_input_producer(
            file_list['image_file_list'])
        random.shuffle(file_list['cam_file_list'])
        cam_paths_queue = tf.train.string_input_producer(
            file_list['cam_file_list'])
        self.steps_per_epoch = int(
            len(file_list['image_file_list'])//self.batch_size)
        # Load images
        img_reader = tf.WholeFileReader()
        _, image_contents = img_reader.read(image_paths_queue)
        image_seq = tf.image.decode_jpeg(image_contents)
        tgt_image, src_image_stack = \
            self.unpack_image_sequence(
                image_seq, self.img_height, self.img_width, self.num_source)

        # Load camera intrinsics
        cam_reader = tf.TextLineReader()
        _, raw_cam_contents = cam_reader.read(cam_paths_queue)
        rec_def = []
        for i in range(9):
            rec_def.append([1.])
        raw_cam_vec = tf.decode_csv(raw_cam_contents, 
                                    record_defaults=rec_def)
        raw_cam_vec = tf.stack(raw_cam_vec)
        intrinsics = tf.reshape(raw_cam_vec, [3, 3])
        
        # Form training batches
        src_image_stack, tgt_image, intrinsics = \
                tf.train.batch([src_image_stack, tgt_image, intrinsics], 
                               batch_size=self.batch_size)
                
        # Data augmentation
        image_all = tf.concat([tgt_image, src_image_stack], axis=3)
        image_all, intrinsics = self.data_augmentation(
            image_all, intrinsics, self.img_height, self.img_width)
        tgt_image = image_all[:, :, :, :3]
        src_image_stack = image_all[:, :, :, 3:]
        intrinsics = self.get_multi_scale_intrinsics(
            intrinsics, self.num_scales)
        
        return tgt_image, src_image_stack, intrinsics, file_list['image_file_list']
        
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

    def depth_augmentation(self, im, intrinsics, out_h, out_w, seed):
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
        return im, intrinsics

    def data_augmentation(self, im, intrinsics, out_h, out_w, seed):
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
        im = tf.cast(im, dtype=tf.uint8)
        return im, intrinsics

    def format_file_list(self, data_root, split, camera):
        with open(data_root + '/%s_%d.txt' % (split, camera), 'r') as f:
            frames = f.readlines()
        subfolders = [x.split(' ')[0] for x in frames]
        frame_ids = [x.split(' ')[1][:-1] for x in frames]
        image_file_list = [os.path.join(data_root, subfolders[i], 
            frame_ids[i] + '.jpg') for i in range(len(frames))]
        cam_file_list = [os.path.join(data_root, subfolders[i], 
            frame_ids[i] + '_cam.txt') for i in range(len(frames))]
        all_list = {}
        all_list['image_file_list'] = image_file_list
        all_list['cam_file_list'] = cam_file_list
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
    
    def unpack_image_sequence_gcnet(self, image_seq, img_height, img_width, num_source, which_img):
        # Implements cases to dynamically slice the sequence tensor, depending on sequence length
        def f0(): return tf.constant(0)
        def f1(): return tf.constant(1)
        def f2(): return tf.constant(2)
        def f3(): return tf.constant(3)
        def f4(): return tf.constant(4)
        img_idx = tf.case({tf.equal(which_img, tf.constant(0)): f0,
            tf.equal(which_img, tf.constant(1)): f1,
            tf.equal(which_img, tf.constant(2)): f2,
            tf.equal(which_img, tf.constant(3)): f3,
            tf.equal(which_img, tf.constant(4)): f4})
        tgt_start_idx = tf.cast(img_width * img_idx, tf.int32)
        print(image_seq.get_shape()) 
        print(tgt_start_idx)
        tgt_image = tf.slice(image_seq, 
                             [0, 0, tgt_start_idx, 0], 
                             [-1, -1, img_width, -1])
        return tgt_image
    
    def load_gcnet_img(self, file_path_left, file_path_right):
        img_seq_left = cv2.imread(file_path_left)
        img_seq_left = cv2.cvtColor(img_seq_left, cv2.COLOR_BGR2RGB)
        img_seq_right = cv2.imread(file_path_right)
        img_seq_right = cv2.cvtColor(img_seq_right, cv2.COLOR_BGR2RGB)
        return img_seq_left, img_seq_right
    
    def load_raw_cam_vec(self, file_path):
        with open(file_path, 'r') as cam:
            reader = csv.reader(cam)
            cam_vec = list(reader)
            raw_cam_vec = np.reshape(np.array(cam_vec), (9,))
        return raw_cam_vec
    
    def shuffle_files(self, all_list_2, all_list_3):
        img_file_list_2 = all_list_2['image_file_list']
        img_file_list_3 = all_list_3['image_file_list']
        cam_file_list_2 = all_list_2['cam_file_list']
        cam_file_list_3 = all_list_3['cam_file_list']
        file_lists = list(zip(img_file_list_2, img_file_list_3, cam_file_list_2, cam_file_list_3))
        random.shuffle(file_lists)
        img_file_list_2_sh, img_file_list_3_sh, cam_file_list_2_sh, cam_file_list_3_sh = zip(*file_lists)
        return img_file_list_2_sh, img_file_list_3_sh, cam_file_list_2_sh, cam_file_list_3_sh

    def augment_new(self, image_seq, raw_cam_vec, depths, seed):
        print('image_seq shape: ' + str(image_seq.get_shape()))
        tgt_image, src_image_stack = \
                self.batch_unpack_image_sequence(
                        image_seq, self.img_height, self.img_width, self.num_source)
        print('tgt_image shape: ' + str(tgt_image.get_shape()))
        print('src_image_stack shape: ' + str(src_image_stack.get_shape()))

        # Get number of channels
#        num_channels = tgt_image.get_shape()[-1].value
#        print('Number of channels: ' + str(num_channels))

        # Load camera intrinsics
        intrinsics = tf.reshape(raw_cam_vec, [3, 3])
        intrinsics = tf.expand_dims(intrinsics, 0)
        print('Shape of intrinsics: ' + str(intrinsics.get_shape()))
        print('Shape of depth: ' + str(depths.get_shape()))
        # Augment depth
        depths_augmented = []
        for i in range(self.num_source + 1):
            depth_exp = tf.expand_dims(tf.expand_dims(depths[i,:,:], 0), -1)
            depth_aug, _ = self.depth_augmentation(depth_exp, intrinsics, self.img_height, self.img_width, seed)
            print('depth_aug size: ' + str(depth_aug[:,:,:,0].get_shape()))
            depths_augmented.append(depth_aug[:,:,:,0])
        depths_augmented = tf.concat(depths_augmented,0) 

        # Data augmentation
        image_all = tf.concat([tgt_image, src_image_stack], axis=3)
        image_all, intrinsics = self.data_augmentation(
            image_all, intrinsics, self.img_height, self.img_width, seed)
        tgt_image = image_all[:, :, :, :3]
        src_image_stack = image_all[:, :, :, 3:]
        intrinsics = self.get_multi_scale_intrinsics(
            intrinsics, self.num_scales)
        
        # To define epoch size
        file_list = self.format_file_list(self.dataset_dir, 'train', 2)
        self.steps_per_epoch = int(
            len(file_list['image_file_list'])//1)
        return tgt_image, src_image_stack, intrinsics, depths

    def scale_down(self, tensor, factor):
        print(tensor)
        h = tensor.get_shape()[0].value
        w = tensor.get_shape()[1].value
        out_h = tf.cast(h/factor, dtype=tf.int32)
        out_w = tf.cast(w/factor, dtype=tf.int32)
        # second possibility is to use tf.image.resize_area()
        tensor_exp = tf.expand_dims(tf.expand_dims(tensor,0),-1)
        tensor_scaled = tf.image.resize_area(tensor_exp,[out_h,out_w])
        tensor_scaled = tf.squeeze(tensor_scaled)
        return tensor_scaled
    
    def process_img_seq(self, image_seq, raw_cam_vec, depth):
        seed = random.randint(1,2**31 - 1)
        # Expand depth tensor from shape [h, w] to [1,h,w,1], so it can be augmented
        print(depth.get_shape())
#        depth = tf.expand_dims(tf.expand_dims(depth, 0), -1)
        # Augment data
        tgt_image_2, src_image_stack_2, intrinsics_2, depth_augm = self.augment_new(image_seq, raw_cam_vec, depth, seed)
        # Scale and recast depth to float
        depth_augm = tf.cast(tf.squeeze(depth_augm), dtype=tf.float32)
        tgt_depth_augm_scaled = []
        tgt_depth_augm_scaled.append(depth_augm[self.num_source//2])
        tgt_depth_augm_scaled.append(self.scale_down(tgt_depth_augm_scaled[0], 2))
        tgt_depth_augm_scaled.append(self.scale_down(tgt_depth_augm_scaled[1], 2))
        tgt_depth_augm_scaled.append(self.scale_down(tgt_depth_augm_scaled[2], 2))

        return tgt_image_2, src_image_stack_2, intrinsics_2, depth_augm, tgt_depth_augm_scaled
    
    # stack depths as fourth channel on RGB images, depths_all is a stack of all the full scale depth estimations
    def stack_rgbd(self, tgt_img, src_img_stack, depths_all):
        # Unpack image depths
#        tgt_img, src_img_stack = self.batch_unpack_image_sequence(image_all, self.img_height, self.img_width, self.num_source)
        depth_tgt_img = depths_all[0,:,:]
        depth_src_img_all = depths_all[1:,:,:]
        print('depths shape (stack rgbd): ' + str(depths_all.get_shape()))
        # Stack target imame depth on RGB image
        depth_tgt_img = tf.expand_dims(tf.expand_dims(depth_tgt_img, 0), -1)
        tgt_img_rgbd = tf.concat([tgt_img, depth_tgt_img], axis=3)
        
        # stack source images
        src_img_stack_rgbd = []
        for src in range(self.num_source):
            src_img = tf.slice(src_img_stack, 
                                 [0, 0, 0, src*3], 
                                 [-1, -1, -1, 3])
            depth_src_img = depth_src_img_all[src,:,:]
            depth_src_img = tf.expand_dims(tf.expand_dims(depth_src_img, 0), -1)
            src_img_rgbd = tf.concat([src_img, depth_src_img], axis=3)
            src_img_stack_rgbd.append(src_img_rgbd)
        src_img_stack_rgbd = tf.concat(src_img_stack_rgbd, axis=3)
        print('Stacked tgt_img shape: ' + str(tgt_img_rgbd.get_shape()))
        print('Stacked src_images shape: ' + str(src_img_stack_rgbd.get_shape()))

        return tgt_img_rgbd, src_img_stack_rgbd, tf.squeeze(depth_tgt_img)
