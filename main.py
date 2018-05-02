import matplotlib

matplotlib.use('agg')

import argparse
import os
import sys
from time import time

from data import load_data
from model import ModelConfig, build_model

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf


class Timer(object):
    """
    with Timer("name"): to make timing easier
    """

    def __init__(self, name=None, output=True):
        self.name = name
        self.output = output
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time()
        return self

    def __exit__(self, *args):
        self.end = time()
        self.elapsed = self.end - self.start
        if self.output:
            print(self)

    def __str__(self):
        return "{}: {:0.6f} seconds".format(self.name, self.elapsed) if self.name else "{:0.6f} seconds".format(
            self.elapsed)


def main(args):
    config_args = {arg: val for arg, val in vars(args).items() if val is not None}
    configs = ModelConfig(**config_args)

    is_training = tf.placeholder(tf.bool)

    x_train, x_test, y_train, y_test = load_data(args.datadir)

    in_tensors = tf.placeholder(tf.float32, shape=(None, 64, 64, 3))
    in_classes = tf.placeholder(tf.float32, shape=(None, args.num_classes))
    model = build_model(in_tensors, configs, is_training)

    with tf.name_scope('predictions'):
        out_y_pred = tf.nn.softmax(model)
    with tf.name_scope('loss_score'):
        loss_score = tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=in_classes)
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(loss_score)
    tf.summary.scalar('loss', loss)
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(args.learn_rate).minimize(loss)

    output_dir = args.outputdir
    tb_log_dir = os.path.join(output_dir, 'tensorboard_logs')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(tb_log_dir):
        os.mkdir(tb_log_dir)

    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(tb_log_dir)
    with tf.Session() as session:
        summary_writer.add_graph(session.graph)
        session.run(tf.global_variables_initializer())

        with Timer("Training"):
            for epoch in range(args.epochs):
                with Timer("Epoch #{}".format(epoch)):
                    print("Epoch=", epoch)
                    tf_score = []

                    for mb in minibatcher(x_train, y_train, args.batch_size, shuffle=True):
                        tf_output = session.run([optimizer, loss, merged],
                                                feed_dict={in_tensors: mb[0],
                                                           in_classes: mb[1],
                                                           is_training: True})

                        summary_writer.add_summary(tf_output[2], global_step=epoch)
                        tf_score.append(tf_output[1])
                    print(" train_loss_score=", np.mean(tf_score))

        # after the training is done, time to test it on the test set
        print("TEST SET PERFORMANCE")
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        y_test_pred, test_loss = session.run([out_y_pred, loss],
                                             feed_dict={in_tensors: x_test,
                                                        in_classes: y_test,
                                                        is_training: False},
                                             options=run_options,
                                             run_metadata=run_metadata)

        summary_writer.add_run_metadata(run_metadata=run_metadata, tag='predict')

        print(" test_loss_score=", test_loss)
        y_test_pred_classified = np.argmax(y_test_pred, axis=1).astype(np.int32)
        y_test_true_classified = np.argmax(y_test, axis=1).astype(np.int32)
        print(classification_report(y_test_true_classified, y_test_pred_classified))

        cm = confusion_matrix(y_test_true_classified, y_test_pred_classified)

        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion.png'))

        # And the log2 version, to emphasize the misclassifications
        plt.imshow(np.log2(cm + 1), interpolation='nearest', cmap=plt.get_cmap("tab20"))
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_log.png'))


def minibatcher(X, y, batch_size, shuffle):
    assert X.shape[0] == y.shape[0]
    n_samples = X.shape[0]

    if shuffle:
        idx = np.random.permutation(n_samples)
    else:
        idx = list(range(n_samples))

    for k in range(int(np.ceil(n_samples / batch_size))):
        from_idx = k * batch_size
        to_idx = (k + 1) * batch_size
        yield X[idx[from_idx:to_idx], :, :, :], y[idx[from_idx:to_idx], :]


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--nconv', dest='nconv', metavar='n', type=int, help='Number of convolutional layers')
    parser.add_argument('-f', '--filters', dest='nfilters', metavar='n', type=int, nargs='+',
                        help='Number of filters [at each layer]')
    parser.add_argument('-k', '--kernel', dest='kernel', metavar='n', type=int, nargs='+',
                        help='Size of the filters [at each layer]')
    parser.add_argument('-m', '--maxpool', dest='maxpool', metavar='n', type=int, nargs='+',
                        help='Maxpooling samples [at each layer]')
    parser.add_argument('-d', '--dropout', dest='dropout', metavar='d', type=float, nargs='+',
                        help='Dropout rate [at each layer]')
    parser.add_argument('-n', '--neurons', dest='fc_units', metavar='n', type=int, nargs='+',
                        help='Number of neurons in the fully connected layer')
    parser.add_argument('-l', '--numclasses', dest='num_classes', metavar='n', type=int,
                        help='Number of classes in the data set')
    parser.add_argument('-r', '--learn_rate', dest='learn_rate', metavar='d', type=float, default=0.001)
    parser.add_argument('-b', '--batch_size', dest='batch_size', metavar='s', type=int, default=256)
    parser.add_argument('-e', '--epochs', dest='epochs', metavar='n', type=int, default=256)
    parser.add_argument('-a', '--activation', dest='activation', choices=['r', 'lr'], default='lr')
    parser.add_argument('datadir', default='dataset')
    parser.add_argument('outputdir', default='out')

    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
