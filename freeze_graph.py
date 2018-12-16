import argparse
import os
import sys

from tensorflow.python.framework import graph_util
import tensorflow as tf
from network_conv import inference

def main(args):
    with tf.Graph().as_default():
        sess = tf.Session()

        embeddings_size = 512

        input_placeholder = tf.placeholder(tf.float32, shape=[None, embeddings_size], name='input')

        net, gender_logits, age_logits = inference(input_placeholder, [], [], training=False)

        gender = tf.argmax(tf.nn.softmax(gender_logits), 1, name="gender")
        
        age_ = tf.cast(tf.constant([i for i in range(0, 117)]), tf.float32)
        age__ = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)
        age = tf.cast(age__, tf.int64, name="age")
        # below for openvino
        # gender = tf.nn.softmax(gender_logits, name="gender")
        # age = tf.nn.softmax(age_logits, name="age")
        # output = tf.concat([age, gender], axis=1, name="output")        
        
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
        # below for openvino, 1 output
        # output_graph_def = freeze_graph_def(sess, input_graph_def, 'output')

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
        if (node.name.startswith('Net') or
                node.name.startswith('Logits') or node.name.startswith('input')):
            whitelist_names.append(node.name)

    # Replace all the variables in the graph with constants of the same values
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, input_graph_def, output_node_names.split(","),
        variable_names_whitelist=whitelist_names)
    return output_graph_def

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_dir', type=str, 
        help='Directory containing the metagraph (.meta) file and the checkpoint (ckpt) file containing model parameters')
    parser.add_argument('--output_file', type=str, 
        help='Filename for the exported graphdef protobuf (.pb)')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
