import argparse
import os
import glob
import cv2
import numpy as np
import tensorflow as tf
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

def run(images, model_path):
    graph = load_graph(model_path)

    with tf.Session(graph=graph) as sess:
        images_pl = graph.get_tensor_by_name('input:0')
        age = graph.get_tensor_by_name('age:0')
        gender = graph.get_tensor_by_name('gender:0')

        # when you freeze graph, it should include nodes that perform
        # image standardization

        return sess.run([age, gender], feed_dict={
           images_pl: images,
        })

def load_images(image_dir):
    addrs = np.array(glob.glob(image_dir))

    aligned_images = []
    file_paths = []

    for i in range(len(addrs)):
        # cv2 load images as BGR
        # just keep it as BGR, when we input to network, it gets converted
        image = cv2.imread(addrs[i])
        # print('Read image:', addrs[i])
        # smooth the image
        kernel = np.ones((3,3), np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        # # sharpen a little
        image = cv2.addWeighted(image, 1.5, image, -0.5, 0)
        # assume image is already aligned
        image = cv2.resize(image, (160, 160), interpolation = cv2.INTER_CUBIC)
        aligned_images.append(image)
        file_paths.append(addrs[i])
    
    return aligned_images, file_paths

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="./demo/faces/*.jpg", help="Path to JPEG images")
    parser.add_argument("--model_path", type=str, default="./models/frozen.pb", help="Model path")
    parser.add_argument("--cuda", default=False, action="store_true",
                        help="Set this flag will use cuda when testing.")
    args = parser.parse_args()
    if not args.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    aligned_images, file_paths = load_images(args.image_dir)
    ages, genders = run(aligned_images, args.model_path)

    for i in range(len(file_paths)):
        gender = "F" if genders[i] == 1 else "M"
        print(file_paths[i] + ': gender=' + gender, ' age=', ages[i])
