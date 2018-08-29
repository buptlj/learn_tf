import tensorflow as tf
import mnist
import os

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_step', 1200, 'Number of steps to run trainer')
tf.app.flags.DEFINE_string('train_dir', './train', 'Directory where to write event logs and checkpoint')


def main():
    train_images, train_labels = mnist.inputs(['./train_img.tfrecords'], mnist.TRAIN_EXAMPLES_NUM,
                                              batch_size=8, shuffle=True)
    train_op, loss, pred = mnist.model_slim(train_images, train_labels)
    saver = tf.train.Saver()
    step = 1
    with tf.Session() as sess:
        init_op = tf.group(
            tf.local_variables_initializer(),
            tf.global_variables_initializer())
        sess.run(init_op)
        ckpt = os.path.join(FLAGS.train_dir, 'model.ckpt')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        while step <= FLAGS.max_step:
            _, train_loss, predict, label = sess.run([train_op, loss, pred, train_labels])
            if step % 100 == 0:
                print('step: {}, loss: {}'.format(step, train_loss))
                saver.save(sess, ckpt, global_step=step)
            step += 1

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
