import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow import keras

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('image_height', 28, 'the height of image')
tf.app.flags.DEFINE_integer('image_width', 28, 'the width of image')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Number of images to process in a batch')
TRAIN_EXAMPLES_NUM = 55000
VALIDATION_EXAMPLES_NUM = 5000
TEST_EXAMPLES_NUM = 10000


def parse_data(example_proto):
    features = {'img_raw': tf.FixedLenFeature([], tf.string, ''),
                'label': tf.FixedLenFeature([], tf.int64, 0)}
    parsed_features = tf.parse_single_example(example_proto, features)
    image = tf.decode_raw(parsed_features['img_raw'], tf.uint8)
    label = tf.cast(parsed_features['label'], tf.int64)
    image = tf.reshape(image, [FLAGS.image_height, FLAGS.image_width, 1])
    image = tf.cast(image, tf.float32)
    return image, label


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
                                            capacity=min_queue_examples + batch_size * 3)
        return images, labels


def inference(images, training):
    with tf.variable_scope('conv1'):
        conv1 = tf.layers.conv2d(inputs=images,
                                 filters=32,
                                 kernel_size=[5, 5],
                                 padding='same',
                                 activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)      # 14*14*32

    with tf.variable_scope('conv2'):
        conv2 = tf.layers.conv2d(inputs=pool1,
                                 filters=64,
                                 kernel_size=[5, 5],
                                 padding='same',
                                 activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)      # 7*7*64

    with tf.variable_scope('fc1'):
        pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
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
    num_batches_per_epoch = TRAIN_EXAMPLES_NUM / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * 10)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(learning_rate=0.001,
                                    global_step=global_step,
                                    decay_steps=decay_steps,
                                    decay_rate=0.1,
                                    staircase=True)

    # opt = tf.train.GradientDescentOptimizer(lr)
    # opt = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.99)
    opt = tf.train.AdamOptimizer(learning_rate=lr)
    grad = opt.compute_gradients(total_loss)
    apply_grad_op = opt.apply_gradients(grad, global_step)

    return apply_grad_op


def model_slim(images, labels):
    net = slim.conv2d(images, 32, [5, 5], scope='conv1')
    net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool1')
    net = slim.conv2d(net, 64, [5, 5], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool2')
    net = slim.flatten(net, scope='flatten')
    net = slim.fully_connected(net, 1024, scope='fully_connected1')
    logits = slim.fully_connected(net, 10, activation_fn=None, scope='fully_connected2')

    prob = slim.softmax(logits)
    loss = slim.losses.sparse_softmax_cross_entropy(logits, labels)
    global_step = tf.train.get_or_create_global_step()
    train_op = train(loss, global_step)

    return train_op, loss, prob


def model_fn(features, labels, mode):
    with tf.variable_scope('conv1'):
        conv1 = tf.layers.conv2d(inputs=features,
                                 filters=32,
                                 kernel_size=[5, 5],
                                 padding='same',
                                 activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)      # 14*14*32

    with tf.variable_scope('conv2'):
        conv2 = tf.layers.conv2d(inputs=pool1,
                                 filters=64,
                                 kernel_size=[5, 5],
                                 padding='same',
                                 activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)      # 7*7*64

    with tf.variable_scope('fc1'):
        pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
        fc1 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        dropout1 = tf.layers.dropout(inputs=fc1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    with tf.variable_scope('logits'):
        logits = tf.layers.dense(inputs=dropout1, units=10)     # 使用该值计算交叉熵损失
        predict = tf.nn.softmax(logits)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()
        train_op = train(loss, global_step)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"eval_accuracy": accuracy}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def input_fn(filenames, training):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_data)

    if training:
        dataset = dataset.shuffle(buffer_size=50000)
    dataset = dataset.batch(FLAGS.batch_size)
    if training:
        dataset = dataset.repeat()

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


def model_keras():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters=32,
                                  kernel_size=[5, 5],
                                  padding='same',
                                  activation=tf.nn.relu,
                                  input_shape=[FLAGS.image_height, FLAGS.image_width, 1]))
    model.add(keras.layers.MaxPool2D(pool_size=[2, 2], strides=2))
    model.add(keras.layers.Conv2D(filters=64,
                                  kernel_size=[5, 5],
                                  padding='same',
                                  activation=tf.nn.relu))
    model.add(keras.layers.MaxPool2D(pool_size=[2, 2], strides=2))
    model.add(keras.layers.Flatten(input_shape=[7, 7, 64]))
    model.add(keras.layers.Dense(units=1024, activation=tf.nn.relu))
    model.add(keras.layers.Dropout(rate=0.4))
    model.add(keras.layers.Dense(units=10))
    model.add(keras.layers.Activation(tf.nn.softmax))

    opt = keras.optimizers.Adam(0.001)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
