import argparse
import os
import glob
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

def run(features, model_path):
    graph = load_graph(model_path)

    with tf.Session(graph=graph) as sess:
        input = graph.get_tensor_by_name('input:0')
        age = graph.get_tensor_by_name('age:0')
        gender = graph.get_tensor_by_name('gender:0')

        return sess.run([age, gender], feed_dict={
           input: features,
        })

def load_embeddings(image_dir):
    addrs = np.array(glob.glob(image_dir))

    embeddings = []
    file_paths = []

    for i in range(len(addrs)):
        embeddings.append(load_csv_features(addrs[i]))
        file_paths.append(addrs[i])
    
    return embeddings, file_paths

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="./demo/embeddings/*.csv", help="Features file pattern")
    parser.add_argument("--model_path", type=str, default="./models/frozen.pb", help="Model path")
    parser.add_argument("--cuda", default=False, action="store_true",
                        help="Set this flag will use cuda when testing.")
    args = parser.parse_args()
    if not args.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    embeddings, file_paths = load_embeddings(args.features)
    ages, genders = run(embeddings, args.model_path)

    for i in range(len(file_paths)):
        gender = "F" if genders[i] == 1 else "M"
        print(file_paths[i] + ': gender=' + gender, ' age=', ages[i])
