import tensorflow as tf
import mnist


def main():
    train_images, train_labels = mnist.inputs(['./train_img.tfrecords'], mnist.TRAIN_EXAMPLES_NUM,
                                              batch_size=8, shuffle=True)
    train_op, loss, pred = mnist.model_slim(train_images, train_labels)

    step = 1
    with tf.Session() as sess:
        init_op = tf.group(
            tf.local_variables_initializer(),
            tf.global_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        while step <= 1000:
            _, train_loss, predict, label = sess.run([train_op, loss, pred, train_labels])
            # print(step, train_loss, '\n', predict, '\n', label)
            if step % 100 == 0:
                print(step, train_loss, '\n', predict, '\n', label)

            step += 1

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    main()