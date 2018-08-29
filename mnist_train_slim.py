import tensorflow as tf
import mnist
import os
from tensorflow.contrib import slim

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_step', 12000, 'Number of steps to run trainer')
tf.app.flags.DEFINE_string('train_dir', './train', 'Directory where to write event logs and checkpoint')

tf.logging.set_verbosity(tf.logging.INFO)


def main():
    dataset = tf.data.TFRecordDataset(['./train_img.tfrecords'])
    dataset = dataset.map(mnist.parse_data)
    dataset = dataset.shuffle(buffer_size=50000)
    dataset = dataset.batch(FLAGS.batch_size)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    train_images, train_labels = iterator.get_next()

    train_op, loss, pred = mnist.model_slim(train_images, train_labels)
    train_tensor = slim.learning.create_train_op(loss, train_op)
    result = slim.learning.train(train_tensor, FLAGS.train_dir, number_of_steps=FLAGS.max_step, log_every_n_steps=100)
    print(result)


if __name__ == '__main__':
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    main()
