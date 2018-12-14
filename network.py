import tensorflow as tf
import tensorflow.contrib.slim as slim

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

def inference(features, age_labels, gender_labels, training=True):
	# Define inputs
	inputs = tf.layers.batch_normalization(
		inputs=features, axis=1,
		momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
		scale=True, training=training, fused=True)
	inputs = tf.nn.relu(inputs)

	net = slim.fully_connected(inputs, 1024, activation_fn=None,
															scope='fc1_1', reuse=False)
	net = slim.fully_connected(net, 512, activation_fn=None,
															scope='fc1_2', reuse=False)

	###
	net = slim.fully_connected(net, 1024, activation_fn=None,
															scope='fc2_1', reuse=False)
	net = slim.fully_connected(net, 512, activation_fn=None,
															scope='fc2_2', reuse=False)

	net = slim.dropout(net, 0.5, is_training=training, scope='Dropout')

	###
	net = slim.fully_connected(net, 1024, activation_fn=None,
															scope='fc3_1', reuse=False)
	net = slim.fully_connected(net, 512, activation_fn=None,
															scope='fc3_2', reuse=False)

	###
	net = slim.fully_connected(net, 1024, activation_fn=None,
															scope='fc4_1', reuse=False)
	net = slim.fully_connected(net, 512, activation_fn=None,
															scope='fc4_2', reuse=False)
	
	###
	net = slim.fully_connected(net, 256, activation_fn=None,
															scope='Bottleneck', reuse=False)

	gender_logits = slim.fully_connected(net, 2, activation_fn=None,
																	 weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
																	 weights_regularizer=slim.l2_regularizer(1e-5),
																	 scope='Logits_Age', reuse=False)
	age_logits = slim.fully_connected(net, 117, activation_fn=None,
																weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
																weights_regularizer=slim.l2_regularizer(1e-5),
																scope='Logits_Gender', reuse=False)

	return net, gender_logits, age_logits
