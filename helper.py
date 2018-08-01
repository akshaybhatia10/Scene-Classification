import tensorflow as tf
import tensorflow_hub as hub

def check_count(num_classes):
	"""Check if number of classes in dataset
	Args:
		num_classes: number of classes for classification
	Return:
		True if 2 or more classes found	
	"""
	if num_classes == 0:
		print ('No classes(folders)!!')
		return 0
	elif num_classes == 1:
		print ('Found 1 class(folder). Atleast 2 required.')
		return 0
	else:
		print ('Found {} classes !!'.format(num_classes))
		return True		

def record_ops(op):
	"""Records summaries for various operation
	Args:
		op: operation for which summary is to be recored			
	"""
	with tf.name_scope('ops'):
		mean = tf.reduce_mean(op)
		std = tf.sqrt(tf.reduce_mean(tf.square(op - mean)))
		min_op = tf.reduce_min(op)
		max_op = tf.reduce_max(op)
		tf.summary.scalar('mean', mean)
		tf.summary.scalar('standard_deviation', std)
		tf.summary.scalar('min', min_op)
		tf.summary.scalar('max', max_op)
		tf.summary.histogram('histgram', op)

def decode_and_resize(hub_module):
	"""Performs image processing steps(decoding and reshaping)
	Args:
		hub_module: Tensorflow Hub module
	Returns:
		placeholder for image data
		reshaped tensor as expected by graph
	"""
	module = hub.load_module_spec(hub_module)
	h, w = hub.get_expected_image_size(module)
	reshape_specs = tf.stack((h, w))
	num_channels = hub.get_num_image_channels(module)
	
	data_placeholder = tf.placeholder(tf.string, name='data_placeholder')
	decode = tf.image.decode_jpeg(data_placeholder, channels=num_channels)
	decode = tf.image.convert_image_dtype(decode, tf.float32)
	decode = tf.expand_dims(decode, 0)
	reshape = tf.cast(reshape_specs, dtype=tf.int32)
	reshaped_image = tf.image.resize_bilinear(decode, reshape)

	return  data_placeholder, reshaped_image
