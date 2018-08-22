import tensorflow as tf
from tensorflow.contrib import slim

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('image_height', 28, 'the height of image')
tf.app.flags.DEFINE_integer('image_width', 28, 'the width of image')

TRAIN_EXAMPLES_NUM = 55000
VALIDATION_EXAMPLES_NUM = 5000
TEST_EXAMPLES_NUM = 10000


def read_mnist_tfrecords(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, features={
        'img_raw': tf.FixedLenFeature([], tf.string, ''),
        'label': tf.FixedLenFeature([], tf.int64, 0)
    })
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int64)
    image = tf.reshape(image, [FLAGS.image_height, FLAGS.image_width, 1])
    return image, label


def inputs(filenames, examples_num, batch_size, shuffle):
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    with tf.name_scope('inputs'):
        filename_queue = tf.train.string_input_producer(filenames)
        image, label = read_mnist_tfrecords(filename_queue)
        image = tf.cast(image, tf.float32)
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(min_fraction_of_examples_in_queue * examples_num)
        num_process_threads = 16
        if shuffle:
            images, labels = tf.train.shuffle_batch([image, label], batch_size=batch_size,
                                                    num_threads=num_process_threads,
                                                    capacity=min_queue_examples + batch_size * 3,
                                                    min_after_dequeue=min_queue_examples)
        else:
            images, labels = tf.train.batch([image, label], batch_size=batch_size,
                                            num_threads=num_process_threads,
                                            capacity=min_queue_examples + batch_size * 3,
                                            min_after_dequeue=min_queue_examples)
        return images, labels


def inference(images, training):
    with tf.variable_scope('conv1'):
        conv1 = tf.layers.conv2d(inputs=images,
                                 filters=32,
                                 kernel_size=[5, 5],
                                 padding='same',
                                 activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    with tf.variable_scope('conv2'):
        conv2 = tf.layers.conv2d(inputs=pool1,
                                 filters=64,
                                 kernel_size=[5, 5],
                                 padding='same',
                                 activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    with tf.variable_scope('fc1'):
        pool2_flat = tf.reshape(pool2, [images.get_shape().as_list()[0], -1])
        fc1 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        dropout1 = tf.layers.dropout(inputs=fc1, rate=0.4, training=training)

    with tf.variable_scope('logits'):
        logits = tf.layers.dense(inputs=dropout1, units=10)     # 使用该值计算交叉熵损失
        predict = tf.nn.softmax(logits)

    return logits, predict


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy')
    cross_entropy_loss = tf.reduce_mean(cross_entropy)
    return cross_entropy_loss


def train(total_loss, global_step):
    # opt = tf.train.GradientDescentOptimizer(0.01)
    # opt = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.99)
    opt = tf.train.AdamOptimizer(learning_rate=0.001)
    grad = opt.compute_gradients(total_loss)
    apply_grad_op = opt.apply_gradients(grad, global_step)

    return apply_grad_op


def model_slim(images, labels):
    net = slim.conv2d(images, 48, [5, 5], scope='conv1')
    net = slim.max_pool2d(net, [2, 2], scope='pool1')
    net = slim.conv2d(net, 96, [5, 5], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    net = slim.flatten(net, scope='flatten')
    net = slim.fully_connected(net, 512, scope='fully_connected1')
    logits = slim.fully_connected(net, 10, activation_fn=None, scope='fully_connected2')

    prob = slim.softmax(logits)
    loss = slim.losses.sparse_softmax_cross_entropy(logits, labels)
    train_op = slim.optimize_loss(loss, slim.get_global_step(),
                                  learning_rate=0.001,
                                  optimizer='Adam')

    return train_op, loss, prob
