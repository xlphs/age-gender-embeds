import glob
import numpy as np
import sys
import os
from random import shuffle
import tensorflow as tf
from utils import load_csv_features

def read_all(data_path):
	addrs = np.array(glob.glob(data_path))
	age_labels = np.array([(addr.split('_')[0].split('/')[1]) for addr in addrs])
	gender_labels = np.array([addr.split('_')[1] for addr in addrs])

	return [addrs, age_labels, gender_labels]

# convert to tensorflow function
def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _floats_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def save_to_tfrecords(outpath, addrs, age_labels, gender_labels):
	writer = tf.python_io.TFRecordWriter(outpath)

	for i in range(len(addrs)):
		embeddings = load_csv_features(addrs[i])
		age = age_labels[i]
		gender = gender_labels[i]

		# print('age ', age, '  gender', gender, '  embeddings', embeddings)

		# Create a feature
		feature = {'age': _int64_feature(age.astype(np.int8)),
							 'gender': _int64_feature(gender.astype(np.int8)),
							 'features': _floats_feature(embeddings),
							 'file_name': _bytes_feature(os.path.basename(addrs[i].encode()))}

		# Create an example protocol buffer
		example = tf.train.Example(features=tf.train.Features(feature=feature))

		# Serialize to string and write on the file
		writer.write(example.SerializeToString())

	writer.close()
	sys.stdout.flush()

def shuffle_data(data):
	# 0=addrs, 1=age_labels, 2=gender_labels
	print('addresses', data[0].shape)
	print('age_labels', data[1].shape)
	print('gender_labels', data[2].shape)
	c = list(zip(data[0], data[1], data[2]))
	shuffle(c)
	addrs, age_labels, gender_labels = zip(*c)
	return [addrs, age_labels, gender_labels]

if __name__ == '__main__':
	data_path = 'dataset/*.csv'

	data = read_all(data_path)
	[addrs, age_labels, gender_labels] = shuffle_data(data)

	# get 2000 for testing, the rest for training
	test_addrs = addrs[0:2000]
	test_ages = age_labels[0:2000]
	test_genders = gender_labels[0:2000]

	train_addrs = addrs[2000:]
	train_ages = age_labels[2000:]
	train_genders = gender_labels[2000:]

	save_to_tfrecords('tfrecords/train.tfrecords', 
		train_addrs, train_ages, train_genders)

	save_to_tfrecords('tfrecords/test.tfrecords', 
		test_addrs, test_ages, test_genders)
