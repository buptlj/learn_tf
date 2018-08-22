import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_step', 1000, 'Number of steps to run trainer')
tf.app.flags.DEFINE_integer('batch_size', 16, 'the width of image')
tf.app.flags.DEFINE_string('train_dir', './train', 'the width of image')


def eval_once(saver, top_k_op):
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]