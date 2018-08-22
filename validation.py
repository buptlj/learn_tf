import tensorflow as tf
import mnist
import numpy as np
import time

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('eval_interval_secs', 100, 'How often to run the eval')
tf.app.flags.DEFINE_integer('batch_size', 100, 'Number of images to process in a batch')
tf.app.flags.DEFINE_string('train_dir', './train', 'Directory where to write event logs and checkpoint')
tf.app.flags.DEFINE_boolean('run_once', True, 'whether to run eval only once')


def eval_once(saver, top_k_op):
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('no checkpoint file')
            return

        coord = tf.train.Coordinator()
        try:
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            iter_per_epoch = mnist.VALIDATION_EXAMPLES_NUM / FLAGS.batch_size

            total_sample = iter_per_epoch * FLAGS.batch_size
            correct_predict = 0
            step = 0

            while step < iter_per_epoch and not coord.should_stop():
                predict = sess.run(top_k_op)
                correct_predict += np.sum(predict)
                step += 1

            precision = correct_predict / total_sample
            print('step: {}, model: {}, precision: {}'.format(global_step, ckpt.model_checkpoint_path, precision))

        except Exception as e:
            print('exception: ', e)
            coord.request_stop(e)
        finally:
            coord.request_stop()
        coord.join(threads)


def evaluation():
    images, labels = mnist.inputs(['./validation_img.tfrecords'], mnist.VALIDATION_EXAMPLES_NUM,
                                  batch_size=FLAGS.batch_size, shuffle=False)
    logits, pred = mnist.inference(images, training=False)
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    saver = tf.train.Saver()

    while True:
        eval_once(saver, top_k_op)
        if FLAGS.run_once:
            break
        time.sleep(FLAGS.eval_interval_secs)


evaluation()
