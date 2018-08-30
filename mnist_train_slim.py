import tensorflow as tf
import mnist
from tensorflow.contrib import slim

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_step', 12000, 'Number of steps to run trainer')
tf.app.flags.DEFINE_string('train_dir', './train', 'Directory where to write event logs and checkpoint')

tf.logging.set_verbosity(tf.logging.INFO)


def train():
    train_images, train_labels = mnist.input_fn(['./train_img.tfrecords'], True)

    train_op, loss, pred = mnist.model_slim(train_images, train_labels, is_training=True)
    train_tensor = slim.learning.create_train_op(loss, train_op)
    result = slim.learning.train(train_tensor, FLAGS.train_dir, number_of_steps=FLAGS.max_step, log_every_n_steps=100)
    print('final step loss: {}'.format(result))


if __name__ == '__main__':
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()
