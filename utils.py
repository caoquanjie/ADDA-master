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



def conv2d(x,w):
  return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x,ksize=[1,3,3,1],
                        strides=[1,2,2,1],padding='SAME')


def fresh_dir(path):
    if tf.gfile.Exists(path):
        tf.gfile.DeleteRecursively(path)
    tf.gfile.MakeDirs(path)


