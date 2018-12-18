import tensorflow as tf
import tensorflow.contrib.slim as slim

_BATCH_NORM_DECAY = 0.995
_BATCH_NORM_EPSILON = 0.001

def inference(features, age_labels, gender_labels, training=True):
	# inputs.shape = [?, 512]

	net = slim.fully_connected(features, 512,
															scope='Net/Bottleneck', reuse=False)

	net = slim.dropout(net, 0.8, is_training=training, scope='Dropout')

	branch1 = slim.fully_connected(net, 768, scope='Net/Branch1/fc1', reuse=False)
	branch1 = slim.dropout(branch1, 0.64, is_training=training, scope='Dropout')
	age_logits = slim.fully_connected(branch1, 117, activation_fn=None,
																weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
																weights_regularizer=slim.l2_regularizer(1e-5),
																scope='Logits/Age', reuse=False)

	branch2 = slim.fully_connected(net, 768, scope='Net/Branch2/fc1', reuse=False)
	branch2 = slim.dropout(branch2, 0.8, is_training=training, scope='Dropout')
	gender_logits = slim.fully_connected(branch2, 2,
																	 weights_initializer=slim.initializers.xavier_initializer(),
																	 weights_regularizer=slim.l2_regularizer(1e-5),
																	 scope='Logits/Gender', reuse=False)

	return net, gender_logits, age_logits

def transfer(facenet_output, features, age_labels, gender_labels, training=True, weight_decay=1e-5):
  branch1 = slim.fully_connected(facenet_output, 768, scope='Net/Branch1/fc1', reuse=False)
  branch1 = slim.dropout(branch1, 0.64, is_training=training, scope='Dropout')
  age_logits = slim.fully_connected(branch1, 117, activation_fn=None,
                                    weights_initializer=slim.initializers.xavier_initializer(),
                                    weights_regularizer=slim.l2_regularizer(weight_decay),
                                    scope='Net/Logits/Age', reuse=False)

  branch2 = slim.fully_connected(facenet_output, 768, scope='Net/Branch2/fc1', reuse=False)
  branch2 = slim.dropout(branch2, 0.8, is_training=training, scope='Dropout')
  gender_logits = slim.fully_connected(branch2, 2,
                                       weights_initializer=slim.initializers.xavier_initializer(),
                                       weights_regularizer=slim.l2_regularizer(weight_decay),
                                       scope='Net/Logits/Gender', reuse=False)

  return gender_logits, age_logits
