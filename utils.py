import tensorflow as tf
import os

def load_csv_features(path):
    with open(path, 'r') as file:
        csv = file.read()
        # turn into array of floats
        return [float(i) for i in csv.split(',')]

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    embeddings_size = 512

    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'age': tf.FixedLenFeature([], tf.int64),
            'gender': tf.FixedLenFeature([], tf.int64),
            'features': tf.FixedLenFeature([embeddings_size], tf.float32),
            'file_name': tf.FixedLenFeature([], tf.string)
        })

    embeddings = features['features']
    age = features['age']
    gender = features['gender']
    file_path = features['file_name']
    return embeddings, age, gender, file_path

def get_inputs(path, batch_size, num_epochs, allow_smaller_final_batch=True):
    if not num_epochs: num_epochs = None

    filename_queue = tf.train.string_input_producer([path], num_epochs=num_epochs)

    image, age, gender, file_path = read_and_decode(filename_queue)

    images, sparse_labels, genders, file_paths = tf.train.batch(
        [image, age, gender, file_path], batch_size=batch_size)

    return images, sparse_labels, genders, file_paths

def losses(logits, labels):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    mean_loss = tf.reduce_mean(loss)
    return mean_loss
