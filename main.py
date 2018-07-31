import argparse
#import tensorflow as tf
from dataset import load_dataset
from helper import check_count
from model import build_graph

if __name__ == '__main__':
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
	print (args)
	# Load dataset and get all files
	files = load_dataset(args.data_dir, args.test_size, args.val_size)
	num_classes = len(files)
	if check_count(num_classes):
		build_graph(args.hub_module)
	else:
		exit()
