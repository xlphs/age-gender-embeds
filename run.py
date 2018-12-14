import argparse
import os

import numpy as np
import tensorflow as tf
from network import inference
from utils import get_inputs, load_csv_features

def run(image_path, model_path):
    with tf.Graph().as_default():
        sess = tf.Session()
        
        input_placeholder = tf.placeholder(tf.float32, shape=[None, 512], name='input')
        features = np.array(load_csv_features(image_path)).reshape(1, 512)

        net, gender_logits, age_logits = inference(input_placeholder, [], [], training=False)

        gender = tf.argmax(tf.nn.softmax(gender_logits), 1)
        
        age_ = tf.cast(tf.constant([i for i in range(0, 117)]), tf.float32)
        age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)
        
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        
        sess.run(init_op)
        
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("restored!")
        else:
            pass

        return sess.run([age, gender], feed_dict={input_placeholder: features})

def main(model_path, image_path, batch_size):
    best_gender_acc, gender_model, best_age_mae, age_model, result = choose_best_model(model_path, image_path,
                                                                                       batch_size)
    return best_gender_acc, gender_model, best_age_mae, age_model, result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="./test.csv", help="Features path")
    parser.add_argument("--model_path", type=str, default="./models/", help="Model path")
    parser.add_argument("--cuda", default=False, action="store_true",
                        help="Set this flag will use cuda when testing.")
    args = parser.parse_args()
    if not args.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    age, gender = run(args.features, args.model_path)

    print('age ', age)
    print('gender ', "F" if gender == 1 else "M")
