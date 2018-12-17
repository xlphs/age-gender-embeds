import argparse
import os
import time

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import inception_resnet_v1
from network_conv import transfer

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    embeddings_size = 512

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

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer([path], num_epochs=num_epochs)

        image, age, gender, file_path = read_and_decode(filename_queue)

        images, sparse_labels, genders, file_paths = tf.train.batch(
            [image, age, gender, file_path], batch_size=batch_size)

        return images, sparse_labels, genders, file_paths


def test_once(tfrecords_path, batch_size, model_checkpoint_path):
    with tf.Graph().as_default():
        sess = tf.Session()

        phase_train = False
        features, age_labels, gender_labels, file_paths = get_inputs(path=tfrecords_path,
                                                    batch_size=batch_size, num_epochs=1)

        # Build the inference graph
        # see facenet train_softmax.py
        prelogits, _ = inception_resnet_v1.inference(features, keep_probability=0.8, 
                phase_train=phase_train, bottleneck_layer_size=512, 
                weight_decay=5e-4)
        net, gender_logits, age_logits = transfer(prelogits,
                features, age_labels, gender_labels, phase_train, 5e-4)

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
        prob_age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)
        abs_age_error = tf.losses.absolute_difference(prob_age, age_labels)

        prob_gender = tf.argmax(tf.nn.softmax(gender_logits), 1)
        gender_acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(gender_logits, gender_labels, 1), tf.float32))

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        sess.run(init_op)
        saver = tf.train.Saver()
        saver.restore(sess, model_checkpoint_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        mean_error_age, mean_gender_acc, mean_loss = [], [], []
        try:
            while not coord.should_stop():
                prob_gender_val, real_gender, prob_age_val, real_age, image_val, gender_acc_val, abs_age_error_val, cross_entropy_mean_val, file_names = sess.run(
                    [prob_gender, gender_labels, prob_age, age_labels, features, gender_acc, abs_age_error, total_loss,
                     file_paths])
                mean_error_age.append(abs_age_error_val)
                mean_gender_acc.append(gender_acc_val)
                mean_loss.append(cross_entropy_mean_val)
                print("Age_MAE:%.2f, Gender_Acc:%.2f%%, Loss:%.2f" % (
                    abs_age_error_val, gender_acc_val * 100, cross_entropy_mean_val))
        except tf.errors.OutOfRangeError:
            print('Summary:')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
        coord.join(threads)
        sess.close()
        return prob_age_val, real_age, prob_gender_val, real_gender, image_val, np.mean(mean_error_age), np.mean(
            mean_gender_acc), np.mean(mean_loss), file_names

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfrecords", type=str, default="./tfrecords/test.tfrecords", help="Testset path")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--model_path", type=str, default="./models/", help="Model path")
    parser.add_argument("--cuda", default=False, action="store_true",
                        help="Set this flag will use cuda when testing.")
    args = parser.parse_args()
    if not args.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    ckpt = tf.train.get_checkpoint_state(args.model_path)
    if ckpt and ckpt.model_checkpoint_path:
        _, _, _, _, _, mean_error_age, mean_gender_acc, mean_loss, _ = test_once(args.tfrecords,
                                                                                 args.batch_size,
                                                                                 ckpt.model_checkpoint_path)
        print("Age_MAE:%.2f, Gender_Acc:%.2f%%, Loss:%.2f" % (mean_error_age, mean_gender_acc * 100, mean_loss))
    else:
        raise IOError("Pretrained model not found!")
