#import tensorflow as tf
import numpy as np
import sys
import os

ALLOWED_EXTENTIONS = ['jpg', 'jpeg', 'JPG', 'JPEG']

def load_dataset(data_dir, test_size, val_size):
	"""
	Args:
		data_dir: path to folder
		test_size: test set percentage
		val_size: val set percentage

	Returns:
		dict containing mapping from from class to training, validation
		and testing set
	"""
	tot = test_size + val_size
	train_size = 100 - tot

	assert test_size >= 1, 'Test percent must be non-negative' 
	assert val_size >= 1, 'Valid percent must be non-negative'
	assert test_size <= 25, 'Keep test percent below 25' 
	assert val_size <= 25, 'Keep valid percent below 25'
	assert tot <= 40, 'Train on atleast 60%. Current training percent {}'.format(train_size)

	if os.path.exists(data_dir):
		dataset = {}
		print ('/{} exists'.format(data_dir))
		folders = [folder for folder in os.listdir(data_dir) if not folder == '.DS_Store']
		print (folders)
		for folder in folders:
			files = []
			files = [file for file in os.listdir(data_dir+'/'+folder) if not file == '.DS_Store']
			num_files = len(files)
			
			shuffled = np.random.permutation(num_files)
			n_val, n_test = int((val_size/100) * num_files), int((test_size/100) * num_files) 
			valid_idx, test_idx, train_idx = shuffled[:n_val], shuffled[n_val:n_val+n_test], shuffled[n_val+n_test:]
			
			print ('{} has {} images'.format(folder, num_files))
			
			train_set, test_set, valid_set = [], [], []
			
			train_set = list(np.squeeze(list(np.take(files, train_idx))))
			test_set = list(np.squeeze(list(np.take(files, test_idx))))
			valid_set = list(np.squeeze(list(np.take(files, valid_idx))))

			dataset[folder] = {'train':train_set, 'valid':valid_set, 'test':test_set}	
		
		return folders, dataset	
	print ('Path does not exist!!')	
	
	return None  