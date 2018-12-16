import tensorflow as tf
import tensorflow.contrib.slim as slim

_BATCH_NORM_DECAY = 0.995
_BATCH_NORM_EPSILON = 0.001

def inference(features, age_labels, gender_labels, training=True):
	# inputs.shape = [?, 512]
	inputs = tf.reshape(features, [-1, 16, 16, 2])

	# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim
	# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/layers.py
	net = slim.conv2d(inputs, 128, 3, scope='Net/Conv2d_1')
	net = slim.max_pool2d(net, 3, stride=2, padding='VALID', scope='Net/MaxPool')

	net = slim.dropout(net, 0.5, is_training=training, scope='Dropout')
	net = slim.flatten(net)

	net = tf.layers.batch_normalization(
		inputs=net, axis=1,
		momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
		scale=True, training=training, name='Net/BatchNorm', fused=True)
	net = tf.nn.relu(net)

	net = slim.fully_connected(net, 512, activation_fn=None,
															scope='Net/Bottleneck', reuse=False)

	net = slim.dropout(net, 0.8, is_training=training, scope='Dropout')

	branch1 = slim.fully_connected(net, 256, activation_fn=None,
															scope='Net/Branch1', reuse=False)
	branch1 = slim.dropout(branch1, 0.8, is_training=training, scope='Dropout')
	branch1 = slim.fully_connected(branch1, 128, activation_fn=None,
															scope='Net/Branch1_2', reuse=False)
	age_logits = slim.fully_connected(branch1, 117, activation_fn=None,
																weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
																weights_regularizer=slim.l2_regularizer(1e-5),
																scope='Logits/Age', reuse=False)

	branch2 = slim.fully_connected(net, 64, activation_fn=None,
															scope='Net/Branch2', reuse=False)
	branch2 = slim.dropout(branch2, 0.8, is_training=training, scope='Dropout')
	branch2 = slim.fully_connected(branch1, 32, activation_fn=None,
															scope='Net/Branch2_2', reuse=False)
	gender_logits = slim.fully_connected(branch2, 2, activation_fn=None,
																	 weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
																	 weights_regularizer=slim.l2_regularizer(1e-5),
																	 scope='Logits/Gender', reuse=False)

	return net, gender_logits, age_logits
