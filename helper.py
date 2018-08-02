import tensorflow as tf
import tensorflow_hub as hub
import os
import numpy as np

SETS = ['train', 'valid', 'test']

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

def find_image_file(files, label, i, path, s):
	"""Find image file
	Args:
		files: Training file names
		label: class for the image file
		i: counter for an image file
		FEATURES_DIR: Path to the store variables
		s: one of - train, valid or test
	Returns:
		Path to image file
	"""
	file_lists = files[label]
	set_lists = file_lists[s]
	assert label in files, 'Class {} not found'.format(label)
	assert s in file_lists, 'Incorrect set category! Must be either train, test, valid'

	idx = i % len(set_lists)
	base = set_lists[idx]
	image_file_path = os.path.join(path, label, base)
	return image_file_path

def find_feature_file(files, label, i, FEATURES_DIR, s, hub_module):
	"""Finds feature (TFHUB) file
	Args:
		files: Training file names
		label: class for the image file
		i: counter for an image file
		FEATURES_DIR: Path to the store variables
		s: one of - train, valid or test
		hub_module: Tensorflow Hub module
	Returns:
		Path to feature file
	"""
	hub_module = hub_module.replace('://', '~').replace('/', '~') \
				.replace(':', '~').replace('\\', '~')
	image_feature_path = find_image_file(files, label, i, FEATURES_DIR, s) \
						 + '_' + hub_module + '.txt'

	return image_feature_path

def compute_features(sess, data, data_placeholder, reshaped_image, 
					 pre_final_tensor, input_tensor):
	"""Computes pre final tensors(features) for 
	   each image and write to file
	Args:
		sess: Current tensorflow session
		data: raw image data
		data_placeholder: Placeholder for image data
		reshaped_image: Reshaped tensor as expected by graph
		pre_final_tensor: pre_final (bottleneck) tensor
		input_tensor: input tensor (expected image size by graph)
	
	Returns:
		Feature values for each image file
	"""
	reshaped_image = sess.run(reshaped_image, {data_placeholder: data})
	feature_values = sess.run(pre_final_tensor, {input_tensor: reshaped_image})
	feature_values = np.squeeze(feature_values)
	
	return feature_values

def make_feature_file(image_feature_path, files, label, i, 
					  data_dir, s, sess, data_placeholder, 
					  reshaped_image, pre_final_tensor, input_tensor):	
	"""Computes pre final tensors(features) for 
	   each image and write to file
	Args:
		image_feature_path: Path to feature file
		files: Training file names
		label: class for the image file
		i: counter for an image file
		data_dir: Path to the dataset
		s: one of - train, valid or test
		sess: Current tensorflow session
		data_placeholder: Placeholder for image data
		reshaped_image: Reshaped tensor as expected by graph
		pre_final_tensor: pre_final (bottleneck) tensor
		input_tensor: input tensor (expected image size by graph)
	"""
	print ('Computing features at {}'.format(image_feature_path))
	image_file_path = find_image_file(files, label, i, data_dir, s)

	if not os.path.exists(image_file_path):
		print ('{} does not exist'.format(image_file_path))
	data = tf.gfile.FastGFile(image_file_path, 'rb').read()
	try:
		feature_values = compute_features(sess, data, data_placeholder, reshaped_image, 
										  pre_final_tensor, input_tensor)
	except Exception as e:
		raise RuntimeError('Error in file {}: {}'.format(image_file_path, str(e)))

	feature_values = ','.join(str(x) for x in feature_values)		
	with open(image_feature_path, 'w') as f:
		f.write(feature_values)

def log_tensor(sess, files, label, i, data_dir, s, FEATURES_DIR, \
			   data_placeholder, reshaped_image, pre_final_tensor, \
			   input_tensor, hub_module):
	"""Computes pre final tensors(features) for each image
	Args:
		sess: Current Tensorflow session
		files: Training file names
		label: class for the image file
		i: counter for an image file
		data_dir: Path to the dataset
		s: one of - train, valid or test
		FEATURES_DIR: Path to the store variables
		data_placeholder: Placeholder for image data
		reshaped_image: Reshaped tensor as expected by graph
		pre_final_tensor: pre_final (bottleneck) tensor
		input_tensor: input tensor (expected image size by graph)
		hub_module: Tensorflow Hub module
	Return:
		array of pre final features for each image
	"""
	file_list = files[label]
	path = os.path.join(FEATURES_DIR, label)
	if not os.path.exists(path):
		os.makedirs(path)
	image_feature_path = find_feature_file(files, label, i, FEATURES_DIR,
									s, hub_module)
	if not os.path.exists(image_feature_path):
		make_feature_file(image_feature_path, files, label, i, 
						  data_dir, s, sess, data_placeholder, 
						  reshaped_image, pre_final_tensor, input_tensor)
	with open(image_feature_path, 'r') as f:
		features = f.read()
	is_valid = False
	try:
		feature_values = [float(x) for x in features.split(',')]
	except ValueError:
		print ('Feature file corrupt. Computing again !')	
		is_valid = True	
	if is_valid:
		make_feature_file(image_feature_path, files, label, i, 
						  data_dir, s, sess, data_placeholder, 
						  reshaped_image, pre_final_tensor, input_tensor)

		with open(image_feature_path, 'r') as f:
			features = f.read()
		feature_values = [float(x) for x in features.split(',')]
	
	return feature_values

def store_tensors(sess, files, data_dir, FEATURES_DIR, data_placeholder, \
				  reshaped_image, pre_final_tensor, input_tensor, hub_module):
	"""Iterate over train, test, valid set and
	   stores all f to disk
	Args:
		sess: Current Tensorflow session
		files: Training file names
		data_dir: Path to the dataset
		FEATURES_DIR: Path to the store variables
		data_placeholder: Placeholder for image data
		reshaped_image: Reshaped tensor as expected by graph
		pre_final_tensor: pre_final (bottleneck) tensor
		input_tensor: input tensor (expected image size by graph)
		hub_module: Tensorflow Hub module
	"""
	num_saves = 0
	SETS = ['train', 'valid', 'test']
	if not os.path.exists(FEATURES_DIR):
		os.makedirs(FEATURES_DIR)
	for label, data_list in files.items():
		for s in SETS:
			file_list = data_list[s]
			for i, file_name in enumerate(file_list):
				log_tensor(sess, files, label, i, data_dir, s,
					FEATURES_DIR, data_placeholder,reshaped_image,
			  		pre_final_tensor, input_tensor, hub_module)

				num_saves += 1
				if num_saves % 100 == 0:
					print ('Number of files saved: {}'.format(num_saves))		