# Copyright

"""GC-Net data loader.
"""

from __future__ import absolute_import, division, print_function
import os
import tensorflow as tf
import readPFM
import numpy as np
import cv2
import scipy
import collections
def string_length_tf(t):
  return tf.py_func(len, [t], [tf.int64])

class GCDataloader(object):
	"""GC-Net dataloader"""
	def __init__(self, data_path, filenames_file, params, dataset, mode):
		self.data_path = data_path
		self.params = params
		self.dataset = dataset
		self.mode = mode
		self.left_img=[]
		self.label=[]
		self.left_image_batch  = None
		self.right_image_batch = None
		self.label_batch=None

		input_queue = tf.train.string_input_producer([filenames_file], shuffle=False)
		line_reader = tf.TextLineReader()
		_, line = line_reader.read(input_queue)

		split_line = tf.string_split([line]).values

		# we load only one image for test, except if we trained a stereo model
		if mode == 'test':
			left_image_path  = tf.string_join([self.data_path, split_line[0]])
			right_image_path = tf.string_join([self.data_path, split_line[1]])
			left_image  = self.read_image(left_image_path)
			right_image = self.read_image(right_image_path)
		else:
			left_image_path  = tf.string_join([self.data_path, split_line[0]])
			right_image_path = tf.string_join([self.data_path, split_line[1]])
			label_path      = tf.string_join([self.data_path, split_line[2]])
			left_image_o  = self.read_image(left_image_path)
			right_image_o = self.read_image(right_image_path)
			label         = self.read_labels(label_path)
			left_image_o=tf.expand_dims(left_image_o,0)
			right_image_o=tf.expand_dims(right_image_o,0)
			label=tf.concat([label,label,label],2)
			label=tf.expand_dims(label,0)
			gather_input=tf.concat([left_image_o,right_image_o,label],0)
			gather_input=tf.random_crop(gather_input,[3,self.params.height,self.params.width,3])
			left_image_o,right_image_o,label=tf.split(gather_input,3,0)
			left_image_o=tf.squeeze(left_image_o)
			right_image_o=tf.squeeze(right_image_o)
			label=tf.squeeze(label)

			label=tf.slice(label,[0,0,0],[self.params.height,self.params.width,1])

		if mode == 'train':
			# randomly flip images
			#do_flip = tf.random_uniform([], 0, 1)
			#left_image  = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(right_image_o), lambda: left_image_o)
			#right_image = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(left_image_o),  lambda: right_image_o)

			# randomly augment images
			do_augment  = tf.random_uniform([], 0, 1)
			left_image, right_image = tf.cond(do_augment > 0.5, lambda: self.augment_image_pair(left_image_o, right_image_o), lambda: (left_image_o, right_image_o))
			#self.left_img=tf.image.convert_image_dtype(left_image,  tf.uint8)
			left_image=(left_image-0.5)/0.5
			right_image=(right_image-0.5)/0.5

			left_image.set_shape( [None, None, 3])
			right_image.set_shape([None, None, 3])
			label.set_shape([None,None,1])

			# capacity = min_after_dequeue + (num_threads + a small safety margin) * batch_size
			min_after_dequeue = 2048

			capacity = min_after_dequeue + 4 * params.batch_size
			self.left_image_batch, self.right_image_batch,self.label_batch= tf.train.shuffle_batch([left_image, right_image,label],
			params.batch_size, capacity, min_after_dequeue, params.num_threads)

		elif mode == 'test':
			left_image=(left_image-0.5)/0.5
			right_image=(right_image-0.5)/0.5
			#self.left_image_batch=tf.expand_dims(left_image,0)
			#self.right_image_batch=tf.expand_dims(right_image,0)
			left_image.set_shape( [None, None, 3])
			right_image.set_shape([None, None, 3])
			self.left_image_batch, self.right_image_batch= tf.train.batch([left_image,right_image],batch_size=1,num_threads=1)
			# capacity = min_after_dequeue + (num_threads + a small safety margin) * batch_size
			#min_after_dequeue = 2
			#capacity = min_after_dequeue + 4 * params.batch_size
			#self.left_image_batch, self.right_image_batch= tf.train.shuffle_batch([left_image, right_image],params.batch_size, capacity, min_after_dequeue, params.num_threads)

	def augment_image_pair(self, left_image, right_image):
		# randomly shift gamma
		random_gamma = tf.random_uniform([], 0.8, 1.2)
		left_image_aug  = left_image  ** random_gamma
		right_image_aug = right_image ** random_gamma

		# randomly shift brightness
		random_brightness = tf.random_uniform([], 0.5, 2.0)
		left_image_aug  =  left_image_aug * random_brightness
		right_image_aug = right_image_aug * random_brightness

		# randomly shift color
		random_colors = tf.random_uniform([3], 0.8, 1.2)
		white = tf.ones([tf.shape(left_image)[0], tf.shape(left_image)[1]])
		color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=2)
		left_image_aug  *= color_image
		right_image_aug *= color_image

		# saturate
		left_image_aug  = tf.clip_by_value(left_image_aug,  0, 1)
		right_image_aug = tf.clip_by_value(right_image_aug, 0, 1)

		return left_image_aug, right_image_aug

	def read_image(self, image_path):
		# tf.decode_image does not return the image size, this is an ugly workaround to handle both jpeg and png
		path_length = string_length_tf(image_path)[0]
		file_extension = tf.substr(image_path, path_length - 3, 3)
		file_cond = tf.equal(file_extension, 'jpg')

		image  = tf.cond(file_cond, lambda: tf.image.decode_jpeg(tf.read_file(image_path)), lambda: tf.image.decode_png(tf.read_file(image_path)))
		image  = tf.image.convert_image_dtype(image,  tf.float32)
		#image  = tf.image.resize_images(image,  [self.params.height, self.params.width], tf.image.ResizeMethod.AREA)
		return image

	def read_labels(self, image_path):
		# tf.decode_image does not return the image size, this is an ugly workaround to handle both jpeg and png
		path_length = string_length_tf(image_path)[0]
		file_extension = tf.substr(image_path, path_length - 3, 3)
		file_cond = tf.equal(file_extension, 'png')

		image  = tf.image.decode_png(tf.read_file(image_path),dtype=tf.uint16)
		image  = tf.image.convert_image_dtype(image,  tf.float32)
		#image  = tf.image.resize_images(image,  [self.params.height, self.params.width], tf.image.ResizeMethod.AREA)
		#image=image*(self.params.width/1240.0)
		return image
