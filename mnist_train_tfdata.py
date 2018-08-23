import tensorflow as tf
import mnist

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_step', 12000, 'Number of steps to run trainer')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Number of images to process in a batch')
tf.app.flags.DEFINE_string('train_dir', './train', 'Directory where to write event logs and checkpoint')


def train():
    # filenames = tf.placeholder(tf.string, [None])
    dataset = tf.data.TFRecordDataset(['train_img.tfrecords'])
    dataset = dataset.map(mnist.parse_data)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(FLAGS.batch_size)
    dataset = dataset.repeat()

    iterator = dataset.make_one_shot_iterator()

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
        # mon_sess.run(iterator.initializer)
        while not mon_sess.should_stop():
            _, train_loss, step, label = mon_sess.run([train_op, loss, global_step, labels])
            if step % 100 == 0:
                print('step: {}, loss: {}'.format(step, train_loss))


if __name__ == '__main__':
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()

