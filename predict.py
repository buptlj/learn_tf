import tensorflow as tf
import cv2
import mnist
import numpy as np


def pred(filename, train_dir):
    img = cv2.imread(filename, flags=cv2.IMREAD_GRAYSCALE)
    img = tf.cast(img, tf.float32)
    img = tf.reshape(img, [-1, 28, 28, 1])

    logits, predict = mnist.inference(img, training=False)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('no checkpoint file')
            return
        pre = sess.run(predict)
        print('model:{}, file:{}, label: {} ({:.2f}%)'.
              format(ckpt.model_checkpoint_path, filename, np.argmax(pre[0]), np.max(pre[0]) * 100))


if __name__ == '__main__':
    pred('./img_test/2_2098.jpg', './train')
