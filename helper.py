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
	with tf.name_score('ops'):
		mean = tf.reduce_mean(op)
		std = tf.sqrt(tf.reduce_mean(tf.square(op - mean)))
		min_op = tf.reduce_min(op)
		max_op = tf.reduce_max(op)
		tf.summary.scalar('mean', mean)
		tf.summary.scalar('standard_deviation', std)
		tf.summary.scalar('min', min_op)
		tf.summary.scalar('max', max_op)
		tf.summary.histogram('histgram', op)
