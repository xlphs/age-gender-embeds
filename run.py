import argparse
import os

import numpy as np
import tensorflow as tf
from network_conv import inference
from utils import get_inputs, load_csv_features

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="")
    return graph

def run(image_path, model_path):
    graph = load_graph(model_path)

    with tf.Session(graph=graph) as sess:
        input = graph.get_tensor_by_name('input:0')
        age = graph.get_tensor_by_name('age:0')
        gender = graph.get_tensor_by_name('gender:0')

        embeddings_size = 512
        features = np.array(load_csv_features(image_path)).reshape(1, embeddings_size)

        return sess.run([age, gender], feed_dict={
           input: features,
        })

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="./test.csv", help="Features path")
    parser.add_argument("--model_path", type=str, default="./models/frozen.pb", help="Model path")
    parser.add_argument("--cuda", default=False, action="store_true",
                        help="Set this flag will use cuda when testing.")
    args = parser.parse_args()
    if not args.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    age, gender = run(args.features, args.model_path)

    print('age ', age)
    print('gender ', "F" if gender == 1 else "M")
