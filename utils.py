from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.contrib.layers as layers
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
#import skimage
#import skimage.transform
#import skimage.io
from tensorflow.contrib.layers import xavier_initializer
distributions = tf.contrib.distributions
#vgg_para = np.load('vgg16.npy', encoding='latin1').item()

def _parameter_summary(params):
    tf.summary.histogram(params.op.name, params)
    tf.summary.histogram(params.op.name + '/row_norm', tf.reduce_sum(tf.pow(tf.norm(params, axis=(0,1)), 2), axis=1))
    tf.summary.scalar(params.op.name + '/spartisty', tf.nn.zero_fraction(params))

def _parameter_summary_fc(params):
    tf.summary.histogram(params.op.name, params)
    tf.summary.histogram(params.op.name + '/row_norm', tf.pow(tf.norm(params, axis=1), 2))
    tf.summary.scalar(params.op.name + '/spartisty', tf.nn.zero_fraction(params))

def _output_summary(outputs):
    tf.summary.histogram(outputs.op.name + '/outputs', outputs)
    tf.summary.scalar(outputs.op.name + '/outputs_sparsity',
		tf.nn.zero_fraction(outputs))


def avg_pool(bottom, name):
    return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def max_pool(bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def max_pool_4x4(bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME', name=name)


def conv_layer_BN(bottom, name, w_shape, b_shape):
    with tf.variable_scope(name):
        n = w_shape[0]*w_shape[1]*w_shape[3]
        filt = get_conv_filter(name, w_shape,n)

        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

        conv_biases = get_bias(name, b_shape)
        bias = tf.nn.bias_add(conv, conv_biases)

        _parameter_summary_fc(filt)
        _output_summary(bias)

    return bias

def relu_layer(x,name):
    with tf.variable_scope(name):
        h = tf.nn.relu(x,name=name)
        _output_summary(h)

    return h



def batch_normalization_layer(x,axis,phase,name):
    with tf.variable_scope(name):
        h = tf.layers.batch_normalization(x,axis=axis,training=phase,name=name)
    return h

def dropout_layer(x,drop_rate,is_train):
    return tf.layers.dropout(inputs=x,rate=drop_rate,training=is_train)

def ConvBNReLU(bottom,w_shape,b_shape,axis,phase,name):
    with tf.variable_scope(name):
        h = conv_layer_BN(bottom,name=name+'/conv',w_shape=w_shape,b_shape=b_shape)

    with tf.variable_scope(name):
        h = batch_normalization_layer(h,axis=axis,phase=phase,name=name+'/BN')

    with tf.variable_scope(name):
        h = relu_layer(h,name = name+'/ReLU')

    return h


# def conv_layer(bottom, name, w_shape, b_shape):
#     with tf.variable_scope(name):
#         filt = get_conv_filter(name, w_shape)
#
#         conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
#
#         conv_biases = get_bias(name, b_shape)
#         bias = tf.nn.bias_add(conv, conv_biases)
#
#         relu = tf.nn.relu(bias)
#
#         _parameter_summary_fc(filt)
#         _output_summary(relu)
#         return relu


def decoder_layer(bottom, name, w_shape, b_shape):
    with tf.variable_scope(name):
        shape = bottom.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        x = tf.reshape(bottom, [-1, dim])

        weights = get_decoder_weight(name, w_shape)
        biases = get_decoder_biases(name, b_shape)

        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

        _parameter_summary_fc(weights)
        _output_summary(fc)

        return fc


def fc_layer(bottom, name, w_shape, b_shape):
    with tf.variable_scope(name):
        shape = bottom.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        x = tf.reshape(bottom, [-1, dim])

        weights = get_fc_weight(name, w_shape)
        biases = get_fcbias(name, b_shape)

        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

        _parameter_summary_fc(weights)
        _output_summary(fc)

        return fc


def get_decoder_weight(name,shape):
    return tf.Variable(tf.truncated_normal(shape=shape,stddev=0.01),name)

def get_decoder_biases(name,shape):
    return tf.Variable(tf.constant(0.0,shape=shape),name=name)


def get_fcbias(name, shape):
    return tf.get_variable(name + '_b', dtype=tf.float32, shape=shape,
                           initializer=tf.constant_initializer(0))


def get_fc_weight(name, shape):
    return tf.get_variable(name + '_W', dtype=tf.float32, shape=shape,
                           initializer=tf.truncated_normal_initializer(stddev=0.01))


def get_conv_filter(name, shape,n):
    return tf.get_variable(name + '_W', dtype=tf.float32, shape=shape,
                           initializer=tf.truncated_normal_initializer(stddev=0.01))


def get_bias(name, shape):
    return tf.get_variable(name + '_b', dtype=tf.float32, shape=shape,
                           initializer=tf.constant_initializer(0))


def get_trans_fcbias(name, shape):
    return tf.get_variable(name + '_b', dtype=tf.float32, shape=shape,
                           initializer=tf.constant_initializer(0.1))


def get_trans_fc_weight(name, shape):
    return tf.get_variable(name + '_W', dtype=tf.float32, shape=shape,
                           initializer=tf.truncated_normal_initializer(stddev=0.1))

def logsoftmax(x):
    xdev = x - tf.reduce_max(x, 1, keep_dims=True)
    lsm = xdev - tf.log(tf.reduce_sum(tf.exp(xdev), 1, keep_dims=True))
    return lsm


def kl_divergence_with_logit(q_logit, p_logit):
    q = tf.nn.softmax(q_logit)
    qlogq = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(q_logit), 1))
    qlogp = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(p_logit), 1))
    return qlogq - qlogp











def save_train_results(images,feats2,feats3,labels,config,step):
    if not os.path.exists(config.train_result_dir):
        os.mkdir(config.train_result_dir)
    fig = plt.figure(1, figsize=(10, 10))
    for i in range(5):
        plt.subplot(5,11,i*11+1)
        plt.title("label:%d%d%d%d%d" % (labels[i][1], labels[i][2], labels[i][3], labels[i][4], labels[i][5]))
        plt.imshow(images[i])
        for idx in range(5):
            plt.subplot(5, 11, i*11+idx + 2)
            plt.imshow(images[i])
            img = skimage.transform.pyramid_expand(feats2[idx][i, 0, :].reshape(8, 8), upscale=8, sigma=10)
            plt.imshow(img,alpha=0.8)
            plt.subplot(5, 11, i*11+idx + 7)
            plt.imshow(images[i])
            img = skimage.transform.pyramid_expand(feats3[idx][i, 0, :].reshape(4, 4), upscale=16, sigma=10)
            plt.imshow(img, alpha=0.8)

    plt.savefig('{}/{}.png'.format(config.train_result_dir, step), bbox_inches='tight')


def save_test_results(images,feats2,feats3,labels,config,step):
    if not os.path.exists(config.test_result_dir):
        os.mkdir(config.test_result_dir)
    fig = plt.figure(1, figsize=(10, 10))
    for i in range(5):
        plt.subplot(5,11,i*11+1)
        plt.title("label:%d%d%d%d%d" % (labels[i][1],labels[i][2],labels[i][3],labels[i][4],labels[i][5]))
        plt.imshow(images[i])
        for idx in range(5):
            plt.subplot(5, 11, i*11+idx + 2)
            plt.imshow(images[i])
            img = skimage.transform.pyramid_expand(feats2[idx][i, 0, :].reshape(8, 8), upscale=8, sigma=10)
            plt.imshow(img,alpha=0.8)
            plt.subplot(5, 11, i*11+idx + 7)
            plt.imshow(images[i])
            img = skimage.transform.pyramid_expand(feats3[idx][i, 0, :].reshape(4, 4), upscale=16, sigma=10)
            plt.imshow(img, alpha=0.8)

    plt.savefig('{}/{}.png'.format(config.test_result_dir, step), bbox_inches='tight')



def weights_variable(name,shape):
  # initial = tf.truncated_normal(shape, stddev=0.01)
  # return tf.Variable(initial)
  return tf.get_variable(name+'_w',dtype=tf.float32,shape=shape,initializer=tf.truncated_normal_initializer(stddev=0.01))
  # return tf.get_variable(name+'w',dtype=tf.float32,shape=shape,initializer= xavier_initializer())


def biass_variable(name,shape):
  # initial = tf.constant(0.0, shape=shape)
  # return tf.Variable(initial)
  return tf.get_variable(name + '_b', dtype=tf.float32, shape=shape,
                         initializer=tf.constant_initializer(0))

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.01)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.0, shape=shape)
  return tf.Variable(initial)


def loglikelihood(mean_arr, sampled_arr, sigma):
  mu = tf.stack(mean_arr)  # mu = [timesteps, batch_sz, loc_dim]
  sampled = tf.stack(sampled_arr)  # same shape as mu
  gaussian = distributions.Normal(mu, sigma)
  logll = gaussian.log_prob(sampled)  # [timesteps, batch_sz, loc_dim]
  logll = tf.reduce_sum(logll, 2)
  logll = tf.transpose(logll)  # [batch_sz, timesteps]
  return logll

def conv2d(x,w):
  return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x,ksize=[1,3,3,1],
                        strides=[1,2,2,1],padding='SAME')


def fresh_dir(path):
    if tf.gfile.Exists(path):
        tf.gfile.DeleteRecursively(path)
    tf.gfile.MakeDirs(path)


