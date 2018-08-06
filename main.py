import argparse
import tensorflow as tf
from dataset import load_dataset
from helper import check_count, decode_and_resize, store_tensors, sample_random_features
from model import build_graph, build_and_retrain, classify_outputs

if __name__ == '__main__':

	VALID_BATCH_SIZE = 16

	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='dataset', help='Path to the dataset')
	parser.add_argument('--val_size', type=float, default='5', help='Validation set percentage')
	parser.add_argument('--test_size', type=float, default='5', help='Test set percentage')
	parser.add_argument('--learning_rate', type=float, default='0.0001', help='Learning Rate')
	parser.add_argument('--steps', type=int, default='10', help='Number of training steps')
	parser.add_argument('--test_step', type=int, default='1', help='Interval to test the model')
	parser.add_argument('--save_step', type=int, default='1', help='Interval to save tested model')
	parser.add_argument('--batch_size', type=int, default='32', help='Batch size')

	parser.add_argument('--model_dir', type=str, default='models/graph.pb', help='Path to the complete trained model file(.pb graph)')
	parser.add_argument('--step_model_dir', type=str, default='step_model_files', help='Path to store step model file(.pb graph)')
	parser.add_argument('--label_dir', type=str, default='models/labels.txt', help='Path to the label file(.txt file)')
	parser.add_argument('--features_dir', type=str, default='features', help='Path to the precomputed features dir')
	parser.add_argument('--tensorboard_summaries_dir', type=str, default='tensorboard_summaries', help='Path to the tensorboard logs')

	parser.add_argument('--hub_module', type=str, 
		default='https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/feature_vector/1', 
		help='Type of Tensorflow Hub module to be used')
	parser.add_argument('--architecture_type', type=str,
		default='mobilenet_1.0_128_quantized', 
		help='Architecture of module')
	parser.add_argument('--final_tensor_name', type=str, default='final_result', help='name scope of final output')

	args = parser.parse_args()
	# Load dataset and get all files
	classes, files = load_dataset(args.data_dir, args.test_size, args.val_size)
	num_classes = len(classes)
	if check_count(num_classes):
		graph, pre_final_tensor, input_tensor = build_graph(args.hub_module)
		
		with graph.as_default():
			optimizer, loss, pre_final_input_tensor, truth_input_tensor, final_output_tensor = \
			build_and_retrain(num_classes, args.final_tensor_name, pre_final_tensor, args.learning_rate, train=True)
		
		session = tf.Session(graph=graph)
		
		with session as sess:
			sess.run(tf.global_variables_initializer())
			
			data_placeholder, reshaped_image = decode_and_resize(args.hub_module)
			store_tensors(sess, files, args.data_dir, args.features_dir, \
						  data_placeholder, reshaped_image, pre_final_tensor, \
						  input_tensor, args.hub_module)
		
			preds, step = classify_outputs(final_output_tensor, truth_input_tensor)

			merge = tf.summary.merge_all()
			writer = {'train': tf.summary.FileWriter(args.tensorboard_summaries_dir + '/train', sess.graph),
					  'valid': tf.summary.FileWriter(args.tensorboard_summaries_dir + '/validation')}
			saver = tf.train.Saver()
			
			for i in range(args.steps + 1):
				features, labels, _ = sample_random_features(sess, num_classes, files, args.batch_size, 'train',
												args.features_dir, args.data_dir, data_placeholder, reshaped_image, 
												pre_final_tensor, input_tensor, args.hub_module)
				train_summary_op, _ = sess.run([merge, optimizer], 
										 feed_dict={pre_final_input_tensor: features, 
										 truth_input_tensor: labels})
				writer['train'].add_summary(train_summary_op, i)

				if (i % args.test_step) == 0:
					acc, l = sess.run([preds, loss], 
						feed_dict={pre_final_input_tensor: features,
						truth_input_tensor: labels})
					print ('Accuracy {}, Loss {:.2f}, Step {}'.format(acc*100, l, i))

				valid_features, valid_labels, _ = sample_random_features(sess, num_classes, files, VALID_BATCH_SIZE, 'valid',
												  args.features_dir, args.data_dir, data_placeholder, reshaped_image, 
												  pre_final_tensor, input_tensor, args.hub_module)


				valid_summary_op, valid_acc = sess.run([merge, preds], 
										 feed_dict={pre_final_input_tensor: valid_features, 
										 truth_input_tensor: valid_labels})
				writer['valid'].add_summary(valid_summary_op, i)
				print ('Validation Accuracy {}, Step {}'.format(valid_acc*100, i))
	
	else:
		exit()
