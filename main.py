import argparse
import tensorflow as tf
from dataset import load_dataset
from helper import check_count, decode_and_resize, store_tensors
from model import build_graph, build_and_retrain

if __name__ == '__main__':
	INPUT_TENSOR_NAME = 'Placeholder'
	FINAL_TENSOR_NAME = 'final_result'
	FEATURES_DIR = 'features'

	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='dataset', help='Path to the dataset')
	parser.add_argument('--val_size', type=float, default='1', help='Validation set percentage')
	parser.add_argument('--test_size', type=float, default='1', help='Test set percentage')
	parser.add_argument('--learning_rate', type=float, default='0.0001', help='Learning Rate')
	parser.add_argument('--epochs', type=int, default='1000', help='Number of training steps')
	parser.add_argument('--batch_size', type=int, default='16', help='Batch size')

	parser.add_argument('--model_dir', type=str, default='models/graph.pb', help='Path to the trained model file(.pb graph)')
	parser.add_argument('--label_dir', type=str, default='models/labels.txt', help='Path to the label file(.txt file)')
	
	parser.add_argument('--hub_module', type=str, 
		default='https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/feature_vector/1', 
		help='Type of Tensorflow Hub module to be used')
	parser.add_argument('--architecture_type', type=str,
		default='mobilenet_1.0_128_quantized', 
		help='Architecture of module')

	args = parser.parse_args()
	# Load dataset and get all files
	classes, files = load_dataset(args.data_dir, args.test_size, args.val_size)
	num_classes = len(classes)
	if check_count(num_classes):
		graph, pre_final_tensor, input_tensor = build_graph(args.hub_module)
		
		with graph.as_default():
			optimizer, loss, pre_final_input_tensor, truth_input_tensor, final_output_tensor = \
			build_and_retrain(num_classes, FINAL_TENSOR_NAME, pre_final_tensor, args.learning_rate, train=True)
		
		session = tf.Session(graph=graph)
		
		with session as sess:
			sess.run(tf.global_variables_initializer())
			
			data_placeholder, reshaped_image = decode_and_resize(args.hub_module)
			store_tensors(sess, files, args.data_dir, FEATURES_DIR, \
						  data_placeholder, reshaped_image, pre_final_tensor, \
						  input_tensor, args.hub_module)
		
	else:
		exit()
