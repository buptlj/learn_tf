import tensorflow as tf
import mnist
import numpy as np
import os
import math

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_step', 10000, 'Number of steps to run trainer')
tf.app.flags.DEFINE_string('train_dir', './train', 'Directory where to write event logs and checkpoint')


def evaluate(sess, top_k_op, training, examples):
    iter_per_epoch = int(math.ceil(examples / FLAGS.batch_size))
    # total_sample = iter_per_epoch * FLAGS.batch_size
    correct_predict = 0
    step = 0

    while step < iter_per_epoch:
        predict = sess.run(top_k_op, feed_dict={training: False})
        correct_predict += np.sum(predict)
        step += 1

    precision = correct_predict / examples
    return precision


def train():
    filenames = tf.placeholder(tf.string, [None])
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(mnist.parse_data)
    dataset = dataset.shuffle(buffer_size=50000)
    dataset = dataset.batch(FLAGS.batch_size)
    dataset = dataset.repeat()

    iterator = dataset.make_initializable_iterator()

    global_step = tf.train.get_or_create_global_step()
    images, labels = iterator.get_next()
    logits, pred = mnist.inference(images, training=True)
    loss = mnist.loss(logits, labels)
    train_op = mnist.train(loss, global_step)

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_step), tf.train.NanTensorHook(loss)],
        save_checkpoint_steps=100
    ) as mon_sess:
        mon_sess.run(iterator.initializer, feed_dict={filenames: ['train_img.tfrecords']})
        while not mon_sess.should_stop():
            _, train_loss, train_step, label = mon_sess.run([train_op, loss, global_step, labels])
            if train_step % 100 == 0:
                print('step: {}, loss: {}'.format(train_step, train_loss))


def train_and_validation():
    training_dataset = tf.data.TFRecordDataset(['./train_img.tfrecords'])
    validation_dataset = tf.data.TFRecordDataset(['./validation_img.tfrecords'])
    test_dataset = tf.data.TFRecordDataset(['./test_img.tfrecords'])

    training_dataset = training_dataset.map(mnist.parse_data)
    training_dataset = training_dataset.shuffle(50000).batch(FLAGS.batch_size).repeat()
    validation_dataset = validation_dataset.map(mnist.parse_data).batch(FLAGS.batch_size)
    test_dataset = test_dataset.map(mnist.parse_data).batch(FLAGS.batch_size)

    iterator = tf.data.Iterator.from_structure(output_types=training_dataset.output_types,
                                               output_shapes=training_dataset.output_shapes)

    training_init_op = iterator.make_initializer(training_dataset)
    validation_init_op = iterator.make_initializer(validation_dataset)
    test_init_op = iterator.make_initializer(test_dataset)
    images, labels = iterator.get_next()

    training = tf.placeholder(dtype=tf.bool)
    logits, pred = mnist.inference(images, training=training)
    loss = mnist.loss(logits, labels)
    top_k_op = tf.nn.in_top_k(logits, labels, 1)
    global_step = tf.train.get_or_create_global_step()
    train_op = mnist.train(loss, global_step)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(training_init_op)
        print('begin to train!')
        ckpt = os.path.join(FLAGS.train_dir, 'model.ckpt')
        train_step = 0
        while train_step < FLAGS.max_step:
            _, train_loss, step, label = sess.run([train_op, loss, global_step, labels], feed_dict={training: True})
            train_step += 1
            if train_step % 100 == 0:
                saver.save(sess, ckpt, train_step)
                if train_step % 1000 == 0:
                    precision = evaluate(sess, top_k_op, training, mnist.TRAIN_EXAMPLES_NUM)
                    print('step: {}, loss: {}, training precision: {}'.format(train_step, train_loss, precision))
                sess.run(validation_init_op)
                precision = evaluate(sess, top_k_op, training, mnist.VALIDATION_EXAMPLES_NUM)
                print('step: {}, loss: {}, validation precision: {}'.format(train_step, train_loss, precision))
                sess.run(training_init_op)
        sess.run(test_init_op)
        precision = evaluate(sess, top_k_op, training, mnist.TEST_EXAMPLES_NUM)
        print('finally test precision: {}'.format(precision))


if __name__ == '__main__':
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    # train()
    train_and_validation()

