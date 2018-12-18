import argparse
import os
import sys
import glob
import numpy as np
import sys
import os
from random import shuffle
from collections import Counter

import tensorflow as tf
import cv2

def read_all(data_path):
    addrs = np.array(glob.glob(data_path))
    age_labels = np.array([(addr.split('_')[0].split('/')[1]) for addr in addrs])
    gender_labels = np.array([addr.split('_')[1] for addr in addrs])
    race_labels = np.array([addr.split('_')[2] for addr in addrs])

    return [addrs, age_labels, gender_labels, race_labels]

def shuffle_data(data):
    # 0=addrs, 1=age_labels, 2=gender_labels, 3=race_labels
    print ('addresses', data[0].shape)
    print ('age_labels', data[1].shape)
    print ('gender_labels', data[2].shape)
    print ('race_labels', data[3].shape)
    # to shuffle data
    c = list(zip(data[0], data[1], data[2], data[3]))
    shuffle(c)
    addrs, age_labels, gender_labels, race_labels = zip(*c)

    return [addrs, age_labels, gender_labels, race_labels]

# data augmentation to balance the races
def augment_image(image, choice):
    if choice == 0:  # original
        return image
    elif choice == 1: # flip vertically
        return cv2.flip(image, 1)

    elif choice == 2 or choice == 3: # add gaussian noice
        row, col, _ = image.shape

        gaussian = np.random.rand(row, col, 1).astype(np.float32)
        gaussian = np.concatenate((gaussian, gaussian, gaussian), axis = 2)
        image_noised = cv2.addWeighted(image, 0.75, 0.25*gaussian, 0.25, 0)

        if choice == 3: # add noise and flip
            image_noised = cv2.flip(image_noised, 1)

        return image_noised

    else: #rotate
        row, col, _ = image.shape

        M = cv2.getRotationMatrix2D((col/2, row/2), 20, 1.0)
        rotated = cv2.warpAffine(image, M, (col, row))
        rotated = rotated[24:176, 24:176]
        rotated = cv2.resize(rotated, (row, col), interpolation = cv2.INTER_CUBIC)

        return rotated

# load image
def load_image(addr):
    # cv2 load images as BGR, convert it to RGB
    image = cv2.imread(addr)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (160, 160))
    image = image.astype(np.float32)
    return image

# convert to tensorflow function
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def save_to_files(addrs, ages, genders, races, filename):
    writer = tf.python_io.TFRecordWriter(filename)

    print(os.path.basename(addrs[0].encode()))

    for i in range(len(addrs)):
        # print how many images are saved every 1000 images
        if not i % 1000:
            print ('Data: {}/{}'.format(i, len(addrs)))
            sys.stdout.flush()

        # Load the image
        image = load_image(addrs[i])

        age = ages[i]
        gender = genders[i]
        race = races[i]

        if race != '0':
            image = augment_image(image, np.random.randint(4))

        image = image.astype(np.int8)

        # Create a feature
        feature = {'age': _int64_feature(age.astype(np.int8)),
                   'gender': _int64_feature(gender.astype(np.int8)),
                   'image_raw': _bytes_feature(tf.compat.as_bytes(image.tostring())),
                   'file_name': _bytes_feature(os.path.basename(addrs[i].encode()))}

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()


def main(args):
    data = read_all(args.images)
    [addrs, age_labels, gender_labels, race_labels] = shuffle_data(data)
    save_to_files(addrs, age_labels, gender_labels, race_labels, args.tfrecords)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--images', type=str, default='UTKFace/*.jpg',
        help='Glob pattern for images')
    parser.add_argument('--tfrecords', type=str, default='tfrecords/train.tfrecords',
        help='Path to save tfrecords')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
