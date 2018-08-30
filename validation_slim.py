import tensorflow as tf
import mnist
import math
from tensorflow.contrib import slim

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', './train', 'Directory where to write event logs and checkpoint')

tf.logging.set_verbosity(tf.logging.INFO)


def validation():
    validation_images, validation_labels = mnist.input_fn(['./validation_img.tfrecords'], False)
    _, loss, pred = mnist.model_slim(validation_images, validation_labels, is_training=False)
    prediction = tf.argmax(pred, axis=1)

    # Choose the metrics to compute:
    value_op, update_op = tf.metrics.accuracy(validation_labels, prediction)
    num_batchs = math.ceil(mnist.VALIDATION_EXAMPLES_NUM / FLAGS.batch_size)

    print('Running evaluation...')
    # Only load latest checkpoint
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.train_dir)

    metric_values = slim.evaluation.evaluate_once(
        num_evals=num_batchs,
        master='',
        checkpoint_path=checkpoint_path,
        logdir=FLAGS.train_dir,
        eval_op=update_op,
        final_op=value_op)
    print(metric_values)


if __name__ == '__main__':
    validation()
