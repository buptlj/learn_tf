import tensorflow as tf
import os
import mnist

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_step', 1000, 'Number of steps to run trainer')
tf.app.flags.DEFINE_integer('batch_size', 16, 'the width of image')
tf.app.flags.DEFINE_string('train_dir', './train', 'the width of image')


def train():
    images, labels = mnist.inputs(['./train_img.tfrecords'], mnist.TRAIN_EXAMPLES_NUM,
                                  FLAGS.batch_size, shuffle=True)
    global_step = tf.train.get_or_create_global_step()

    logits, pred = mnist.inference(images, training=True)
    # logits, pred = mnist.model(images=images, labels=labels)
    loss = mnist.loss(logits, labels)
    train_op = mnist.train(loss, global_step)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init_op = tf.group(
            tf.local_variables_initializer(),
            tf.global_variables_initializer())
        sess.run(init_op)
        ckpt = os.path.join(FLAGS.train_dir, 'model.ckpt')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord=coord)

        for i in range(1, FLAGS.max_step + 1):
            _, train_loss, predict, label = sess.run([train_op, loss, pred, labels])
            # print(predict, '\n', label)
            if i % 100 == 0:
                print('step: {}, loss: {}'.format(i, train_loss))
                print(predict, '\n', label)
                saver.save(sess, ckpt, global_step=i)

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    train()
