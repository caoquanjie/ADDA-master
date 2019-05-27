import  tensorflow as tf
from classisier import Classifier
from generator import GlimpseNet, LocNet, ContextNet, build_ad_loss, build_ad_loss_v2
import tensorflow as tf
from config import Config
from utils import weights_variable,biass_variable,loglikelihood
from collections import OrderedDict
import os
config = Config()



# phase = tf.placeholder(dtype=tf.bool)
# sample_phase = tf.placeholder(dtype=tf.bool)
# keep_prob = tf.placeholder(dtype=tf.float32)

# def s_encoder(images, reuse=False):
#     with tf.variable_scope("s_encoder", reuse=reuse):
#         conv1_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 20], mean=0, stddev=0.1))
#         conv1_b = tf.Variable(tf.zeros(20))
#         conv1 = tf.nn.conv2d(images, conv1_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
#         conv1 = tf.nn.relu(conv1)
#         pool_1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
#
#         conv2_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 20, 50], mean=0, stddev=0.1))
#         conv2_b = tf.Variable(tf.zeros(50))
#         conv2 = tf.nn.conv2d(pool_1, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
#         conv2 = tf.nn.relu(conv2)
#         pool_2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
#
#         fc1 = tf.contrib.layers.flatten(pool_2)
#
#         fc1_w = tf.Variable(tf.truncated_normal(shape=(5*5*50, 500), mean=0, stddev=0.1))
#         fc1_b = tf.Variable(tf.zeros(500))
#         fc1 = tf.matmul(fc1, fc1_w) + fc1_b
#         fc1 = tf.nn.relu(fc1)
#
#
#         return fc1
# def t_encoder(images, reuse=False):
#     with tf.variable_scope("t_encoder", reuse=reuse):
#         conv1_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 20], mean=0, stddev=0.1))
#         conv1_b = tf.Variable(tf.zeros(20))
#         conv1 = tf.nn.conv2d(images, conv1_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
#         conv1 = tf.nn.relu(conv1)
#         pool_1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
#
#         conv2_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 20, 50], mean=0, stddev=0.1))
#         conv2_b = tf.Variable(tf.zeros(50))
#         conv2 = tf.nn.conv2d(pool_1, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
#         conv2 = tf.nn.relu(conv2)
#         pool_2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
#
#         fc1 = tf.contrib.layers.flatten(pool_2)
#
#         fc1_w = tf.Variable(tf.truncated_normal(shape=(5 * 5 * 50, 500), mean=0, stddev=0.1))
#         fc1_b = tf.Variable(tf.zeros(500))
#         fc1 = tf.matmul(fc1, fc1_w) + fc1_b
#         fc1 = tf.nn.relu(fc1)
#
#         return fc1

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

        # conv3 = tf.layers.conv2d(conv2_pooling,filters=80,kernel_size=(5,5),activation=tf.nn.relu,
        #                           trainable=trainable,name = 'conv3')
        #
        # conv3_pooling = tf.layers.max_pooling2d(conv3, (2, 2), (2, 2), name='pool3')

        flat = tf.layers.flatten(conv2_pooling, name='flat')

        fc1 = tf.layers.dense(flat, 128, activation=tf.nn.relu, trainable=trainable, name='fc1')
        fc2 = tf.layers.dense(fc1, 84, activation=tf.nn.tanh, trainable=trainable, name='fc2')

      


    return fc2


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

        fc1 = tf.layers.dense(flat, 128, activation=tf.nn.relu, trainable=trainable, name='fc1')
        fc2 = tf.layers.dense(fc1, 84, activation=tf.nn.tanh, trainable=trainable, name='fc2')

        

    return fc2

#
# def discriminator(inputs, reuse=False, trainable=True):
#     with tf.variable_scope('discriminator', reuse=reuse):
#         fc1 = tf.layers.dense(inputs, 500, activation=tf.nn.relu, trainable=trainable, name='fc1')
#         fc2 = tf.layers.dense(fc1, 256, activation=tf.nn.relu, trainable=trainable, name='fc2')
#         #fc3 = tf.layers.dense(fc2, 64, activation=tf.nn.relu, trainable=trainable, name='fc3')
#         fc4 = tf.layers.dense(fc2, 1, activation=None, trainable=trainable, name='fc4')
#         return fc4


def discriminator(inputs, reuse=False, trainable=True):
    with tf.variable_scope('discriminator', reuse=reuse):
        fc1 = tf.layers.dense(inputs, 128, activation=tf.nn.relu, trainable=trainable, name='fc1',
                              #kernel_initializer=tf.truncated_normal_initializer(0.01),
                              #bias_initializer=tf.constant_initializer(0)
                              )

        fc2 = tf.layers.dense(fc1, 128, activation=tf.nn.relu, trainable=trainable, name='fc2',
                              #kernel_initializer=tf.truncated_normal_initializer(0.01),
                              #bias_initializer=tf.constant_initializer(0)
                              )

        fc3 = tf.layers.dense(fc2, 2, activation=None, trainable=trainable, name='fc3',
                              #kernel_initializer=tf.truncated_normal_initializer(0.01),
                              #bias_initializer=tf.constant_initializer(0)
                           )

        return fc3


def classifier(inputs, phase,reuse=False,trainable=True):
    with tf.variable_scope('classifier', reuse=reuse):
        # fc1 = tf.layers.dense(inputs, 256, activation=tf.nn.relu, trainable=trainable, name='fc1')
        # fc1_drop = tf.nn.dropout(inputs, keep_prob=keep_prob)
        inputs = tf.layers.dropout(inputs,rate=0.5,training=phase)
        fc = tf.layers.dense(inputs, 10, activation=None, trainable=trainable, name='fc',
                             #kernel_initializer=tf.truncated_normal_initializer(0.1),
                             #bias_initializer=tf.constant_initializer(0)
                             )
    return fc

def build_ad_loss_v2(disc_s, disc_t):
  d_loss = -tf.reduce_mean(tf.log(tf.clip_by_value(disc_s, 0.01, 1)) + tf.log(tf.clip_by_value(1 - disc_t, 0.01, 1)))
  g_loss = -tf.reduce_mean(tf.log(tf.clip_by_value(disc_t, 0.01, 1)))
  return g_loss, d_loss


# def build_ad_loss(disc_s, disc_t):
#     g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_t, labels=tf.ones_like(disc_t))
#     g_loss = tf.reduce_mean(g_loss)
#     d_loss = tf.reduce_mean(
#         tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_s, labels=tf.ones_like(disc_s))) + tf.reduce_mean(
#         tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_t, labels=tf.zeros_like(disc_t)))
#
#     return g_loss, d_loss

def build_ad_loss(disc_s, disc_t):
    g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_t, labels=tf.ones_like(disc_t))
    g_loss = tf.reduce_mean(g_loss)
    d_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_s, labels=tf.ones_like(disc_s))) + tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_t, labels=tf.zeros_like(disc_t)))
    # tf.summary.scalar("g_loss", g_loss)
    # tf.summary.scalar('d_loss', d_loss)
    return g_loss, d_loss


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


