import argparse
import os
import sys

from tensorflow.python.framework import graph_util
import tensorflow as tf
import tensorflow.contrib.slim as slim
import inception_resnet_v1
from network_conv import transfer

def main(args):
    with tf.Graph().as_default():
        sess = tf.Session()

        input_placeholder = tf.placeholder(tf.float32, shape=[None, 160, 160, 3], name='input')

        if args.image_process:
            images = tf.map_fn(lambda frame: tf.reverse_v2(frame, [-1]), input_placeholder)
            input_placeholder = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), images)

        phase_train = False
        wd = 5e-4

        # Build the inference graph
        # see facenet train_softmax.py
        prelogits, _ = inception_resnet_v1.inference(input_placeholder, keep_probability=1.0, 
                phase_train=phase_train, bottleneck_layer_size=512, 
                weight_decay=wd)
        net, gender_logits, age_logits = transfer(prelogits,
                input_placeholder, [], [], phase_train, wd)

        gender = tf.argmax(tf.nn.softmax(gender_logits), 1, name="gender")
        
        age_ = tf.cast(tf.constant([i for i in range(0, 117)]), tf.float32)
        age__ = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)
        age = tf.cast(age__, tf.int64, name="age")
        
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        
        sess.run(init_op)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(args.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("restore and continue training!")
        else:
            pass

        # Retrieve the protobuf graph definition and fix the batch norm nodes
        input_graph_def = sess.graph.as_graph_def()

        # Freeze the graph def
        output_graph_def = freeze_graph_def(sess, input_graph_def, 'age,gender')

        # Serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(args.output_file, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph: %s" % (len(output_graph_def.node), args.output_file))

def freeze_graph_def(sess, input_graph_def, output_node_names):
    for node in input_graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in xrange(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']
    
    # Get the list of important nodes
    whitelist_names = []
    for node in input_graph_def.node:
        if (node.name.startswith('InceptionResnet') or node.name.startswith('Net') or
                node.name.startswith('Logits') or node.name.startswith('input')):
            whitelist_names.append(node.name)

    # Replace all the variables in the graph with constants of the same values
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, input_graph_def, output_node_names.split(","),
        variable_names_whitelist=whitelist_names)
    return output_graph_def

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_dir', type=str, default='./models/',
        help='Directory containing the metagraph (.meta) file and the checkpoint (ckpt) file containing model parameters')
    parser.add_argument('--output_file', type=str, default='./models/frozen.pb',
        help='Filename for the exported graphdef protobuf (.pb)')
    parser.add_argument('--image_process', default=True,
        help='Add image processing nodes')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
