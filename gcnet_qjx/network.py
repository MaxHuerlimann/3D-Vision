from config import Config
import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops

MOVING_AVERAGE_DECAY = 0.99
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00001
CONV_WEIGHT_STDDEV = 0.05
GC_VARIABLES = 'gc_variables'
UPDATE_OPS_COLLECTION = 'gc_update_ops'  # training ops
# DISPARITY = 96
NUM_RES_BLOCK = 8  # totally 8 resnet blocks


# wrapper for 2d convolution op
def computeSoftArgMin(logits):
	h, w, d = logits.get_shape().as_list()
	softmax = tf.nn.softmax(logits)
	disp = tf.range(1, d + 1, 1)
	disp = tf.cast(disp, tf.float32)
	disp_mat = []
	for i in range(w * h):
		disp_mat.append(disp)
	disp_mat = tf.reshape(tf.stack(disp_mat), [h, w, d])
	result = tf.multiply(softmax, disp_mat)
	result = tf.reduce_sum(result, 2)
	return result


def conv_2d(x, c):
	ksize = c['ksize']
	stride = c['stride']
	filters_out = c['conv_filters_out']

	filters_in = x.get_shape()[-1]
	shape = [ksize, ksize, filters_in, filters_out]
	weights = tf.get_variable('weights',
	                          shape=shape,
	                          dtype='float32',
	                          initializer=tf.contrib.layers.xavier_initializer(),
	                          regularizer=tf.contrib.layers.l2_regularizer(CONV_WEIGHT_DECAY),
	                          collections=[tf.GraphKeys.GLOBAL_VARIABLES, GC_VARIABLES],
	                          trainable=True)
	bias = tf.get_variable('bias', [filters_out], 'float32', tf.constant_initializer(0.05, dtype='float'))
	x = tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')
	return tf.nn.bias_add(x, bias)


def conv_3d(x, c):
	ksize = c['ksize']
	stride = c['stride']
	filters_out = c['conv_filters_out']
	filters_in = x.get_shape()[-1]
	shape = [ksize, ksize, ksize, filters_in, filters_out]

	weights = tf.get_variable('weights',
	                          shape=shape,
	                          dtype='float32',
	                          initializer=tf.contrib.layers.xavier_initializer(),
	                          regularizer=tf.contrib.layers.l2_regularizer(CONV_WEIGHT_DECAY),
	                          collections=[tf.GraphKeys.GLOBAL_VARIABLES, GC_VARIABLES],
	                          trainable=True)
	bias = tf.get_variable('bias', [filters_out], 'float32', tf.constant_initializer(0.05, dtype='float'))
	x = tf.nn.conv3d(x, weights, [1, stride, stride, stride, 1], padding='SAME')
	return tf.nn.bias_add(x, bias)


def deconv_3d(x, c):
	ksize = c['ksize']
	stride = c['stride']
	filters_out = c['conv_filters_out']
	filters_in = x.get_shape()[-1]

	# must have as_list to get a python list!
	x_shape = x.get_shape().as_list()
	depth = x_shape[1] * stride
	height = x_shape[2] * stride
	width = x_shape[3] * stride
	output_shape = [1, depth, height, width, filters_out]
	strides = [1, stride, stride, stride, 1]
	shape = [ksize, ksize, ksize, filters_out, filters_in]

	initializer = tf.contrib.layers.xavier_initializer()
	weights = tf.get_variable('weights',
	                          shape=shape,
	                          dtype='float32',
	                          initializer=tf.contrib.layers.xavier_initializer(),
	                          regularizer=tf.contrib.layers.l2_regularizer(CONV_WEIGHT_DECAY),
	                          collections=[tf.GraphKeys.GLOBAL_VARIABLES, GC_VARIABLES],
	                          trainable=True)
	bias = tf.get_variable('bias', [filters_out], 'float32', tf.constant_initializer(0.05, dtype='float32'))
	x = tf.nn.conv3d_transpose(x, weights, output_shape=output_shape, strides=strides, padding='SAME')
	return tf.nn.bias_add(x, bias)


# wrapper for batch-norm op
def bn(x, c):
	x_shape = x.get_shape()
	params_shape = x_shape[-1:]

	axis = list(range(len(x_shape) - 1))

	beta = tf.get_variable('beta',
	                       shape=params_shape,
	                       initializer=tf.zeros_initializer(),
	                       dtype='float32',
	                       collections=[tf.GraphKeys.GLOBAL_VARIABLES, GC_VARIABLES],
	                       trainable=True)
	gamma = tf.get_variable('gamma',
	                        shape=params_shape,
	                        initializer=tf.ones_initializer(),
	                        dtype='float32',
	                        collections=[tf.GraphKeys.GLOBAL_VARIABLES, GC_VARIABLES],
	                        trainable=True)

	moving_mean = tf.get_variable('moving_mean',
	                              shape=params_shape,
	                              initializer=tf.zeros_initializer(),
	                              dtype='float32',
	                              collections=[tf.GraphKeys.GLOBAL_VARIABLES, GC_VARIABLES],
	                              trainable=False)
	moving_variance = tf.get_variable('moving_variance',
	                                  shape=params_shape,
	                                  initializer=tf.ones_initializer(),
	                                  dtype='float32',
	                                  collections=[tf.GraphKeys.GLOBAL_VARIABLES, GC_VARIABLES],
	                                  trainable=False)

	# These ops will only be performed when training.
	mean, variance = tf.nn.moments(x, axis)
	update_moving_mean = moving_averages.assign_moving_average(moving_mean,
	                                                           mean, BN_DECAY)
	update_moving_variance = moving_averages.assign_moving_average(
		moving_variance, variance, BN_DECAY)
	tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
	tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

	mean, variance = control_flow_ops.cond(
		c['is_training'], lambda: (mean, variance),
		lambda: (moving_mean, moving_variance))

	x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)

	return x


# resnet block
def stack(x, c):
	shortcut = x
	with tf.variable_scope('block_A'):
		x = conv_2d(x, c)
		x = bn(x, c)
		x = tf.nn.relu(x)
	with tf.variable_scope('block_B'):
		x = conv_2d(x, c)
		x = bn(x, c)
		x = shortcut + x
		x = tf.nn.relu(x)
	return x


# siamese structure
def _build_resnet(x, c):
	with tf.variable_scope('downsample'):
		c['conv_filters_out'] = 32
		c['ksize'] = 5
		c['stride'] = 2
		x = conv_2d(x, c)
		x = bn(x, c)
		x = tf.nn.relu(x)

	c['ksize'] = 3
	c['stride'] = 1

	with tf.variable_scope('resnet'):
		for i in range(NUM_RES_BLOCK):
			with tf.variable_scope('resnet' + str(i + 1)):
				x = stack(x, c)

	c['ksize'] = 5
	c['stride'] = 2
	with tf.variable_scope('resnet_tail'):
		x = conv_2d(x, c)

	return x


def getCostVolume(left_features, right_features, max_d):
	max_d = int(max_d)
	CostVol_left = []
	# CostVol_right=[]
	l_features = tf.squeeze(left_features)
	r_features = tf.squeeze(right_features)
	shape = l_features.get_shape().as_list()
	r_features_pad = tf.pad(r_features, paddings=[[0, 0], [max_d, 0], [0, 0]], mode='CONSTANT')
	# l_features_pad=tf.pad(l_features,paddings=[[0,0],[0,max_d],[0,0]],mode='CONSTANT')
	for d in reversed(range(max_d)):
		left_tensor_slice = l_features
		right_tensor_slice = tf.slice(r_features_pad, [0, d, 0], [shape[0], shape[1], shape[2]])
		CostVol_left.append(tf.concat([left_tensor_slice, right_tensor_slice], 2))
	CostVol_left = tf.stack(CostVol_left)
	# print(CostVol_left)
	CostVol_left = tf.reshape(CostVol_left, [1, max_d, shape[0], shape[1], 2 * shape[2]])
	# for d in range(max_d):
	#	left_tensor_slice=tf.slice(l_features_pad,[0,d,0],[shape[0],shape[1],shape[2]])
	#	right_tensor_slice=r_features
	#	CostVol_right.append(tf.concat([left_tensor_slice,right_tensor_slice],2))
	# CostVol_right=tf.stack(CostVol_right)
	# CostVol_right=tf.reshape(CostVol_right,[1,max_d,shape[0],shape[1],2*shape[2]])

	return CostVol_left


def regular_conv(cost_vol, c):
	c['ksize'] = 3
	c['stride'] = 1
	c['conv_filters_out'] = 32
	with tf.variable_scope('Conv_3d_' + str(19)):
		x = conv_3d(cost_vol, c)
		x = bn(x, c)
		x = tf.nn.relu(x)

	with tf.variable_scope('Conv_3d_' + str(20)):
		x = conv_3d(x, c)
		x = bn(x, c)
		x20 = tf.nn.relu(x)
	print('x20_1' + str(x20.get_shape()))
	c['stride'] = 2
	c['conv_filters_out'] = 64

	with tf.variable_scope('Conv_3d_' + str(21)):
		x = conv_3d(x20, c)
		x = bn(x, c)
		x = tf.nn.relu(x)
	c['stride'] = 1
	with tf.variable_scope('Conv_3d_' + str(22)):
		x = conv_3d(x, c)
		x = bn(x, c)
		x = tf.nn.relu(x)

	with tf.variable_scope('Conv_3d_' + str(23)):
		x = conv_3d(x, c)
		x = bn(x, c)
		x23 = tf.nn.relu(x)
	print('x23_1' + str(x23.get_shape()))

	c['stride'] = 2
	with tf.variable_scope('Conv_3d_' + str(24)):
		x = conv_3d(x23, c)
		x = bn(x, c)
		x = tf.nn.relu(x)

	c['stride'] = 1
	with tf.variable_scope('Conv_3d_' + str(25)):
		x = conv_3d(x, c)
		x = bn(x, c)
		x = tf.nn.relu(x)

	with tf.variable_scope('Conv_3d_' + str(26)):
		x = conv_3d(x, c)
		x = bn(x, c)
		x26 = tf.nn.relu(x)
	print('x26_1' + str(x26.get_shape()))

	c['stride'] = 2
	c['conv_filters_out'] = 128
	with tf.variable_scope('Conv_3d_' + str(27)):
		x = conv_3d(x26, c)
		x = bn(x, c)
		x = tf.nn.relu(x)

	c['stride'] = 1
	with tf.variable_scope('Conv_3d_' + str(28)):
		x = conv_3d(x, c)
		x = bn(x, c)
		x = tf.nn.relu(x)

	with tf.variable_scope('Conv_3d_' + str(29)):
		x = conv_3d(x, c)
		x = bn(x, c)
		x29 = tf.nn.relu(x)
	print('x29_1' + str(x29.get_shape()))

	return x20, x23, x26, x29


def regular_deconv(x20, x23, x26, x, c):
	c['stride'] = 2
	c['conv_filters_out'] = 64
	c['ksize'] = 3
	with tf.variable_scope('Deconv_3d_' + str(30)):
		x = deconv_3d(x, c)
		x = bn(x, c)
		x = tf.nn.relu(x)
		x = x + x26

	print('x30' + str(x.get_shape()))
	with tf.variable_scope('Deconv_3d_' + str(31)):
		x = deconv_3d(x, c)
		x = bn(x, c)
		x = tf.nn.relu(x)
		x = x + x23
	print('x31' + str(x.get_shape()))

	c['conv_filters_out'] = 32
	with tf.variable_scope('Deconv_3d_' + str(32)):
		x = deconv_3d(x, c)
		x = bn(x, c)
		x = tf.nn.relu(x)
		x = x + x20
	print('x32' + str(x.get_shape()))

	with tf.variable_scope('Deconv_3d_' + str(33)):
		x = deconv_3d(x, c)
		x = bn(x, c)
		x = tf.nn.relu(x)
	print('x33' + str(x.get_shape()))

	c['conv_filters_out'] = 1
	with tf.variable_scope('Deconv_3d_' + str(34)):
		x = deconv_3d(x, c)
	return x


def stereo_model(left_x, right_x, params, is_training=False, rv=None):
	c = Config()
	c['is_training'] = tf.convert_to_tensor(is_training,
	                                        dtype='bool',
	                                        name='is_training')
	c['conv_filters_out'] = 32
	with tf.variable_scope('stereo_network', reuse=rv):
		# Unary features
		with tf.variable_scope('siamese') as scope:
			left_features = _build_resnet(left_x, c)
			scope.reuse_variables()
			right_features = _build_resnet(right_x, c)

		# Cost Volume
		CostVol_left = getCostVolume(left_features, right_features, params.max_disparity / 4)

		# 3d convolution
		with tf.variable_scope('3d_conv') as scope_conv:
			lx20, lx23, lx26, lx29 = regular_conv(CostVol_left, c)
		# scope_conv.reuse_variables()
		# rx20,rx23,rx26,rx29,rx32=regular_conv(CostVol_right,c)

		# 3d deconvolution
		with tf.variable_scope('3d_deconv') as scope_deconv:
			left_d = regular_deconv(lx20, lx23, lx26, lx29, c)
		# scope_deconv.reuse_variables()
		# right_d=regular_deconv(rx32,rx20,rx23,rx26,rx29,c)

		left_d = tf.squeeze(left_d)
		left_d = - left_d
		left_d = tf.transpose(left_d, [1, 2, 0])
		disp_pre = computeSoftArgMin(left_d)
	return disp_pre
