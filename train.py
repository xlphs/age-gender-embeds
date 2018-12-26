import argparse
import os
import time

import tensorflow as tf
from network_conv import inference
from utils import get_inputs

def run_training(tfrecords_path, batch_size, epoch, model_path, log_dir, start_lr, wd, kp):
    with tf.Graph().as_default():
        sess = tf.Session()

        features, age_labels, gender_labels, _ = get_inputs(path=tfrecords_path,
                                                    batch_size=batch_size, num_epochs=epoch)

        net, gender_logits, age_logits = inference(features, age_labels, gender_labels,
                                          training=True)
       	
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

        prob_gender = tf.argmax(tf.nn.softmax(gender_logits), 1)
        gender_acc = tf.reduce_mean(tf.to_float(tf.equal(tf.to_int64(prob_gender), gender_labels)))

        tf.summary.scalar("age_cross_entropy", age_cross_entropy_mean)
        tf.summary.scalar("gender_cross_entropy", gender_cross_entropy_mean)
        tf.summary.scalar("total loss", total_loss)
        tf.summary.scalar("train_abs_age_error", abs_loss)
        tf.summary.scalar("gender_accuracy", gender_acc)

        # Add to the Graph operations that train the model.
        global_step = tf.Variable(0, name="global_step", trainable=False)
        lr = tf.train.exponential_decay(start_lr, global_step=global_step,
              decay_steps=2000, decay_rate=0.1, staircase=True)
        optimizer = tf.train.AdamOptimizer(lr)
        tf.summary.scalar("lr", lr)

        # # only train age branch
        # trainable = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Net/Branch1') + tf.get_collection(
        #     tf.GraphKeys.GLOBAL_VARIABLES, scope='Logits/Age')

        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # update batch normalization layer
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(total_loss, global_step) #, var_list=trainable

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(log_dir, sess.graph)

        new_saver = tf.train.Saver(all_vars, max_to_keep=100)
       	ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            new_saver.restore(sess, ckpt.model_checkpoint_path)
            print("restore and continue training!")
        else:
        	pass

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            step = sess.run(global_step)
            start_time = time.time()
            while not coord.should_stop():
                # start_time = time.time()
                # Run one step of the model.  The return values are
                # the activations from the `train_op` (which is
                # discarded) and the `loss` op.  To inspect the values
                # of your ops or variables, you may include them in
                # the list passed to sess.run() and the value tensors
                # will be returned in the tuple from the call.
                _, summary = sess.run([train_op, merged])
                train_writer.add_summary(summary, step)
                # duration = time.time() - start_time
                # Print an overview fairly often.
                if step % 100 == 0:
                    duration = time.time() - start_time
                    print('%.3f sec' % duration)
                    start_time = time.time()
                if step % 1000 == 0:
                    save_path = new_saver.save(sess, os.path.join(model_path, "model.ckpt"), global_step=global_step)
                    print("Model saved in file: %s" % save_path)
                step = sess.run(global_step)
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (epoch, step))
        finally:
            # When done, ask the threads to stop.
            save_path = new_saver.save(sess, os.path.join(model_path, "model.ckpt"), global_step=global_step)
            print("Model saved in file: %s" % save_path)
            coord.request_stop()
        # Wait for threads to finish.
        coord.join(threads)
        sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", "--lr", type=float, default=1e-3, help="Init learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Set 0 to disable weight decay")
    parser.add_argument("--model_path", type=str, default="./models", help="Path to save models")
    parser.add_argument("--log_path", type=str, default="./log", help="Path to save logs")
    parser.add_argument("--epoch", type=int, default=10, help="Epoch")
    parser.add_argument("--tfrecords", type=str, default="./tfrecords/train.tfrecords", help="Path of tfrecords")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size")
    parser.add_argument("--keep_prob", type=float, default=0.8, help="Used by dropout")
    parser.add_argument("--cuda", default=False, action="store_true",
                        help="Set this flag will use cuda when testing.")
    args = parser.parse_args()
    if not args.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    run_training(tfrecords_path=args.tfrecords, batch_size=args.batch_size, epoch=args.epoch, model_path=args.model_path,
                 log_dir=args.log_path, start_lr=args.learning_rate, wd=args.weight_decay, kp=args.keep_prob)
