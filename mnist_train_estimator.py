import tensorflow as tf
import mnist
import os

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_step', 1000, 'Number of steps to run trainer')
tf.app.flags.DEFINE_string('train_dir', './train', 'Directory where to write event logs and checkpoint')

tf.logging.set_verbosity(tf.logging.INFO)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def train():
    my_checkpoint_config = tf.estimator.RunConfig(save_checkpoints_steps=100, keep_checkpoint_max=5)

    mnist_classifier = tf.estimator.Estimator(model_fn=mnist.model_fn, model_dir=FLAGS.train_dir,
                                              config=my_checkpoint_config)
    tensor_to_log = {'probabilities': 'softmax_tensor'}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensor_to_log, every_n_iter=100)

    for i in range(FLAGS.max_step // 100):
        mnist_classifier.train(input_fn=lambda: mnist.input_fn(['./train_img.tfrecords'], True),
                               # hooks=[logging_hook],
                               steps=100)

        eval_results = mnist_classifier.evaluate(input_fn=lambda: mnist.input_fn(['./validation_img.tfrecords'], False))
        print(eval_results)


if __name__ == '__main__':
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()
