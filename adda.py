import tensorflow as tf
from config import Config
import os
config = Config()


# encoder network
def s_encoder(inputs, reuse=False, trainable=True):
    with tf.variable_scope('s_encoder', reuse=reuse):
        inputs = tf.image.resize_images(inputs,[28,28])
        conv1 = tf.layers.conv2d(inputs, filters=20, kernel_size=(5, 5), activation=tf.nn.relu, trainable=trainable,
                                 name='conv1',kernel_initializer=tf.truncated_normal_initializer(0.1),
                                 bias_initializer=tf.constant_initializer(0))

        conv1_pooling = tf.layers.max_pooling2d(conv1, (2, 2), (2, 2), name='pool1')

        conv2 = tf.layers.conv2d(conv1_pooling, filters=50, kernel_size=(5, 5), activation=tf.nn.relu,
                                 trainable=trainable, name='conv2',
                                 kernel_initializer=tf.truncated_normal_initializer(0.1),
                                 bias_initializer=tf.constant_initializer(0)
                                 )

        conv2_pooling = tf.layers.max_pooling2d(conv2, (2, 2), (2, 2), name='pool2')

        flat = tf.layers.flatten(conv2_pooling, name='flat')

        fc1 = tf.layers.dense(flat, 120, activation=tf.nn.relu, trainable=trainable, name='fc1')
        fc2 = tf.layers.dense(fc1, 84, activation=tf.nn.tanh, trainable=trainable, name='fc2')

      
    return fc2


# target network
def t_encoder(inputs, reuse=False, trainable=True):
    with tf.variable_scope('t_encoder', reuse=reuse):
        inputs = tf.image.resize_images(inputs, [28, 28])
        conv1 = tf.layers.conv2d(inputs, filters=20, kernel_size=(5, 5), activation=tf.nn.relu, trainable=trainable,
                                 name='conv1',kernel_initializer=tf.truncated_normal_initializer(0.1),
                                 bias_initializer=tf.constant_initializer(0))

        conv1_pooling = tf.layers.max_pooling2d(conv1, (2, 2), (2, 2), name='pool1')

        conv2 = tf.layers.conv2d(conv1_pooling, filters=50, kernel_size=(5, 5), activation=tf.nn.relu,
                                 trainable=trainable, name='conv2',
                                 kernel_initializer=tf.truncated_normal_initializer(0.1),
                                 bias_initializer=tf.constant_initializer(0)
                                 )

        conv2_pooling = tf.layers.max_pooling2d(conv2, (2, 2), (2, 2), name='pool2')

        flat = tf.layers.flatten(conv2_pooling, name='flat')

        fc1 = tf.layers.dense(flat, 120, activation=tf.nn.relu, trainable=trainable, name='fc1')
        fc2 = tf.layers.dense(fc1, 84, activation=tf.nn.tanh, trainable=trainable, name='fc2')

    return fc2



def discriminator(inputs, reuse=False, trainable=True):
    with tf.variable_scope('discriminator', reuse=reuse):
        fc1 = tf.layers.dense(inputs, 128, activation=tf.nn.relu, trainable=trainable, name='fc1')
        fc2 = tf.layers.dense(fc1, 128, activation=tf.nn.relu, trainable=trainable, name='fc2')
        fc3 = tf.layers.dense(fc2, 2, activation=None, trainable=trainable, name='fc3',)

        return fc3


def classifier(inputs, phase,reuse=False,trainable=True):
    with tf.variable_scope('classifier', reuse=reuse):
        # fc1 = tf.layers.dense(inputs, 256, activation=tf.nn.relu, trainable=trainable, name='fc1')
        # fc1_drop = tf.nn.dropout(inputs, keep_prob=keep_prob)
        inputs = tf.layers.dropout(inputs,rate=0.5,training=phase)
        fc = tf.layers.dense(inputs, 10, activation=None, trainable=trainable, name='fc',
                             )
    return fc

# design a adversarial loss to train discriminator and target network
def concat_ad_loss_2(feat_s, feat_t):
    feat_concat = tf.concat((feat_s, feat_t), 0)
    pre_lab = discriminator(feat_concat, reuse=False, trainable=True)
    src_lab = tf.ones_like(feat_s, tf.int64)[:, 0]
    tgt_lab = tf.zeros_like(feat_t, tf.int64)[:, 0]
    lab_concat = tf.concat((src_lab, tgt_lab), 0)
    d_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pre_lab, labels=lab_concat))
    pre_tgt = discriminator(feat_t, reuse=True, trainable=True)
    g_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pre_tgt, labels=tf.ones_like(pre_tgt, tf.int64)[:, 0]))
    return g_loss, d_loss

def eval(logits, labels):
    pred = tf.nn.softmax(logits)
    correct_label_predicted = tf.equal(labels, tf.argmax(pred, axis=1))
    predicted_accuracy = tf.reduce_mean(tf.cast(correct_label_predicted, tf.float32))
    return predicted_accuracy


