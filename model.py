import tensorflow_hub as hub
import tensorflow as tf
from helper import record_ops, sample_random_features

def build_graph(hub_module):
	"""Build a graph from tensorflow hub module
	
	Args:
		hub_module: Tensorflow Hub module
	Returns:
		graph extracted from hub module
		pre final tensor(bottleneck)
		input tensor (expected image size by graph)	
	"""
	module = hub.load_module_spec(hub_module)
	h, w = hub.get_expected_image_size(module)
	with tf.Graph().as_default() as graph:
		input_tensor = tf.placeholder(tf.float32, shape=(None, h, w, 3))
		mod = hub.Module(module)
		pre_final_tensor = mod(input_tensor)

	return graph, pre_final_tensor, input_tensor

def build_and_retrain(num_classes, final_tensor_name, pre_final_tensor, learning_rate, train):
	"""Add final(and custom) layers to the graph for retraining
	
	Args:
		num_classes: number of classes in our dataset
		final_tensor_name: (constant) name of output of classification(softmax) layer
		pre_final_tensor: pre final(bottleneck) tensor
		train: to train new layers or not
	Returns:

		result of training process tensor - optimizer
		result of loss(cross entropy) process tensor
		pre_final input(bottleneck) tensor
		ground truth tensor
		final output tensor
	"""	
	batch_size, pre_final_tensor_shape = pre_final_tensor.get_shape().as_list()
	assert batch_size is None, 'Batch size should not be constant'
	with tf.name_scope('inputs'):
		pre_final_input_tensor = tf.placeholder_with_default(
								 pre_final_tensor, shape=(batch_size, pre_final_tensor_shape),
								 name='pre_final_input_tensor')
		truth_input_tensor = tf.placeholder(tf.int64, (batch_size), name='truth_input_tensor')

	with tf.name_scope('build_and_retrain'):
		with tf.name_scope('W'):
			w = tf.Variable(tf.truncated_normal(shape=(pre_final_tensor_shape, num_classes), stddev=0.01), name='weights')
			record_ops(w)	

		with tf.name_scope('b'):
			b = tf.Variable(tf.ones((num_classes)), name='bias')
			record_ops(b)

		with tf.name_scope('matmul_op'):
			outputs = tf.add(tf.matmul(pre_final_input_tensor, w), b)
			tf.summary.histogram('pre_classification_output', outputs)		

	final_output_tensor = tf.nn.softmax(outputs, name=final_tensor_name)
	tf.summary.histogram('classification_output', final_output_tensor)
	
	# Check if training or evaluation(inference)	
	if train:
		with tf.name_scope('loss'):
			loss = tf.losses.sparse_softmax_cross_entropy(
				labels=truth_input_tensor, logits=outputs)

		with tf.name_scope('train'):
			optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
		
		return optimizer, loss, pre_final_input_tensor, truth_input_tensor, final_output_tensor

	return None, None, pre_final_input_tensor, truth_input_tensor, final_output_tensor

def classify_outputs(logits, labels):
	"""Calculate prediction scores
	Args:
		logits: Predicted targets
		labels: True targets
	Returns:
		prediction of our network
		evaluation step
	"""
	with tf.name_scope('accuracy'):
		with tf.name_scope('correct_results'):
			preds = tf.argmax(logits, 1)
			correct = tf.equal(preds, labels)
		with tf.name_scope('accuracy'):
			step = tf.reduce_mean(tf.cast(correct, tf.float32))
	tf.summary.scalar('accuracy', step)	

	return preds, step

def compute_test_graph(hub_module, num_classes, final_tensor_name, learning_rate, CHECKPOINT_DIR):
	"""Compute graph from saved session
	Args:
		module: Tensorflow hub module
		num_classes: number of classes in our dataset
		final_tensor_name: name scope of final tensor
		learning_rate: learning rate
		CHECKPOINT_DIR: path to saved model files
	Returns:
		session of tensorflow graph
		final output tensor
		pre_final input(bottleneck) tensor
		ground truth tensor		
		prediction of our network
		evaluation step
	"""
	graph, pre_final_tensor, input_tensor = build_graph(hub_module)
	sess = tf.Session(graph=graph)
	with graph.as_default():
		_, _, pre_final_input_tensor, truth_input_tensor, final_output_tensor = \
			build_and_retrain(num_classes, final_tensor_name, pre_final_tensor, learning_rate, train=False)

	saver = tf.train.Saver()
	saver.restore(sess, CHECKPOINT_DIR)
	pred, step = classify_outputs(final_output_tensor, truth_input_tensor)

	return sess, input_tensor, pre_final_input_tensor, truth_input_tensor, pred, step


def save_file_to_disk(graph, file, hub_module, num_classes, final_tensor_name, learning_rate, CHECKPOINT_DIR):
	"""Saves intermediate model files to disk
	Args:
		graph: tensorflow graph
		file: name of file to be saved
		module: Tensorflow hub module
		num_classes: number of classes in our dataset
	"""
	module = hub.load_module_spec(hub_module)
	sess, _, _, _, _ = compute_test_graph(hub_module, num_classes, final_tensor_name, 
									learning_rate, CHECKPOINT_DIR)
	graph = tf.graph.util.convert_variables_to_constants(
			sess, sess.graph.as_graph_def(), [final_tensor_name])

	with tf.gfile.FastGFile(file, 'wb') as f:
		f.write(graph.SerializeToString())

def compute_final_op(sess, batch_size, features_dir, data_dir, hub_module, num_classes, files, data_placeholder,
				     reshaped_image, pre_final_tensor, input_tensor, final_tensor_name, learning_rate, CHECKPOINT_DIR):
	"""Computes final performance using test set
	Args:
		sess: Current tensorflow session
		batch_size: batch size
		features_dir:
		hub_module: Tensorflow hub module
		num_classes: number of classes in our dataset
		files: Training file names 
		data_placeholder: Placeholder for image data
		reshaped_image: Reshaped tensor as expected by graph
		pre_final_tensor: pre_final (bottleneck) tensor
		input_tensor: input tensor (expected image size by graph)
		final_tensor_name:
		learning_rate:
		CHECKPOINT_DIR:	
	"""
	module = hub.load_module_spec(hub_module)
	features, labels, filenames = sample_random_features(sess, num_classes, files, batch_size, 'test',
						   		  features_dir, data_dir, data_placeholder, reshaped_image, 
						   		  pre_final_tensor, input_tensor, hub_module)

	sess, input_tensor, pre_final_input_tensor, truth_input_tensor, \
								pred, step = compute_test_graph(hub_module, num_classes, 
											 final_tensor_name, learning_rate, CHECKPOINT_DIR)

	acc, p = sess.run([step, pred], feed_dict={data_placeholder: features,
									 truth_input_tensor: labels})

	print ('Final Test Accuracy {}'.format(acc*100))	