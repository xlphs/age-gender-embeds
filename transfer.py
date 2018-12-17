from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
import os
import re
import sys
import time
import tensorflow.contrib.slim as slim
import inception_resnet_v1
from network_conv import transfer

def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'age': tf.FixedLenFeature([], tf.int64),
            'gender': tf.FixedLenFeature([], tf.int64),
            'file_name': tf.FixedLenFeature([], tf.string)
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([160 * 160 * 3])
    image = tf.reshape(image, [160, 160, 3])
    image = tf.reverse_v2(image, [-1]) # FIXME
    image = tf.image.per_image_standardization(image)
    # image = tf.cast(image,tf.float32) * (1. / 255) - 0.5

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    age = features['age']
    gender = features['gender']
    file_path = features['file_name']
    return image, age, gender, file_path

def get_inputs(path, batch_size, num_epochs, allow_smaller_final_batch=True):
    if not num_epochs: num_epochs = None

    filename_queue = tf.train.string_input_producer([path], num_epochs=num_epochs)

    image, age, gender, file_path = read_and_decode(filename_queue)

    images, sparse_labels, genders, file_paths = tf.train.batch(
        [image, age, gender, file_path], batch_size=batch_size)

    return images, sparse_labels, genders, file_paths

def main(args):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model metagraph and checkpoint
            print('Model directory: %s' % args.facenet)
            
            phase_train = True

            features, age_labels, gender_labels, _ = get_inputs(path=args.tfrecords,
                                                    batch_size=args.batch_size, num_epochs=1)

            # Build the inference graph
            # see facenet train_softmax.py
            prelogits, _ = inception_resnet_v1.inference(features, keep_probability=0.8, 
                phase_train=phase_train, bottleneck_layer_size=512, 
                weight_decay=5e-4)
            net, gender_logits, age_logits = transfer(prelogits,
                features, age_labels, gender_labels, phase_train)
            
            # Add to the Graph the loss calculation.
            age_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=age_labels, logits=age_logits)
            age_cross_entropy_mean = tf.reduce_mean(age_cross_entropy)

            gender_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gender_labels,
                                                                                  logits=gender_logits)
            gender_cross_entropy_mean = tf.reduce_mean(gender_cross_entropy)

            # l2 regularization
            total_loss = tf.add_n(
                [gender_cross_entropy_mean, age_cross_entropy_mean] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

            age_ = tf.cast(tf.constant([i for i in range(0, 117)]), tf.float32)
            age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)
            abs_loss = tf.losses.absolute_difference(age_labels, age)

            gender_acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(gender_logits, gender_labels, 1), tf.float32))

            # Train model and update
            tf.summary.scalar("age_cross_entropy", age_cross_entropy_mean)
            tf.summary.scalar("gender_cross_entropy", gender_cross_entropy_mean)
            tf.summary.scalar("train_abs_age_error", abs_loss)
            tf.summary.scalar("gender_accuracy", gender_acc)
            tf.summary.scalar("total loss", total_loss)

            global_step = tf.Variable(0, name="global_step", trainable=False)
            lr = tf.train.exponential_decay(1e-3, global_step=global_step,
                  decay_steps=2000, decay_rate=0.6, staircase=True)
            optimizer = tf.train.AdamOptimizer(lr)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(total_loss, global_step)

            init_op = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
            
            sess.run(init_op)
            print('Init session!')

            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(args.log_dir, sess.graph)

            variables_to_restore = slim.get_variables_to_restore()
            new_saver = tf.train.Saver(variables_to_restore, max_to_keep=5)
            ckpt = tf.train.get_checkpoint_state(args.facenet)
            if ckpt and ckpt.model_checkpoint_path:
                new_saver.restore(sess, ckpt.model_checkpoint_path)
                print("Restored!")
            else:
                pass
            
            save_path = new_saver.save(sess, os.path.join(args.model_path, "model.ckpt"), global_step=global_step)
            print("Model saved in file: %s" % save_path)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="./log", help="Path to save logs")
    parser.add_argument("--facenet", type=str, default="./models/vgg/", help="Path to facenet model")
    parser.add_argument("--model_path", type=str, default="./models", help="Path to save models")
    parser.add_argument("--tfrecords", type=str, default="./tfrecords/train.tfrecords", help="Path of tfrecords")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
