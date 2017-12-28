import tensorflow as tf
import tensorflow.contrib.layers as tcl
import sys

sys.path.insert(0, 'ops/')
from tf_ops import *


'''
   Architecture from https://arxiv.org/pdf/1703.10593.pdf
'''
def netG(x, LOSS_METHOD, INSTANCE_NORM=True, PIXEL_SHUF=False):

   # c7s1-32
   enc_conv1 = tcl.conv2d(x, 32, 7, 1, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_enc_conv1')
   if INSTANCE_NORM: enc_conv1 = instance_norm(enc_conv1)
   else: enc_conv1 = tcl.batch_norm(enc_conv1)
   enc_conv1 = relu(enc_conv1)
   
   # d64
   enc_conv2 = tcl.conv2d(enc_conv1, 64, 3, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_enc_conv2')
   if INSTANCE_NORM: enc_conv2 = instance_norm(enc_conv2)
   else: enc_conv2 = tcl.batch_norm(enc_conv2)
   enc_conv2 = relu(enc_conv2)
   
   # d128
   enc_conv3 = tcl.conv2d(enc_conv2, 128, 3, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_enc_conv3')
   if INSTANCE_NORM: enc_conv3 = instance_norm(enc_conv3)
   else: enc_conv3 = tcl.batch_norm(enc_conv3)
   enc_conv3 = relu(enc_conv3)
   
   print 'x:        ',x
   print 'enc_conv1:',enc_conv1
   print 'enc_conv2:',enc_conv2
   print 'enc_conv3:',enc_conv3
   exit()
   print 'enc_conv2:',enc_conv2
   print 'enc_conv3:',enc_conv3
   print 'enc_conv4:',enc_conv4
   print 'enc_conv5:',enc_conv5
   print 'enc_conv6:',enc_conv6
   print 'enc_conv7:',enc_conv7
   print 'enc_conv8:',enc_conv8
   print

   dec_conv1 = tcl.convolution2d_transpose(enc_conv8, 512, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_dec_conv1')
   dec_conv1 = relu(dec_conv1)
   dec_conv1 = tf.concat([dec_conv1, enc_conv7], axis=3)
   print 'dec_conv1:',dec_conv1

   return dec_conv8


def netD(x, LAYER_NORM, LOSS_METHOD, reuse=False):
   print
   print 'netD'

   sc = tf.get_variable_scope()
   with tf.variable_scope(sc, reuse=reuse):

      conv1 = tcl.conv2d(x, 64, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv1')
      if LOSS_METHOD != 'wgan': conv1 = tcl.batch_norm(conv1)
      elif LOSS_METHOD == 'wgan' and LAYER_NORM: conv1 = tcl.layer_norm(conv1)
      conv1 = lrelu(conv1)
      
      conv2 = tcl.conv2d(conv1, 128, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv2')
      if LOSS_METHOD != 'wgan': conv2 = tcl.batch_norm(conv2)
      elif LOSS_METHOD == 'wgan' and LAYER_NORM: conv2 = tcl.layer_norm(conv2)
      conv2 = lrelu(conv2)
      
      conv3 = tcl.conv2d(conv2, 256, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv3')
      if LOSS_METHOD != 'wgan': conv3 = tcl.batch_norm(conv3)
      elif LOSS_METHOD == 'wgan' and LAYER_NORM: conv3 = tcl.layer_norm(conv3)
      conv3 = lrelu(conv3)
      
      conv4 = tcl.conv2d(conv3, 512, 4, 1, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv4')
      if LOSS_METHOD != 'wgan': conv4 = tcl.batch_norm(conv4)
      elif LOSS_METHOD == 'wgan' and LAYER_NORM: conv4 = tcl.layer_norm(conv4)
      conv4 = lrelu(conv4)
      
      conv5 = tcl.conv2d(conv4, 1, 1, 1, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv5')
      if LOSS_METHOD != 'wgan': conv5 = tcl.batch_norm(conv5)
      elif LOSS_METHOD == 'wgan' and LAYER_NORM: conv5 = tcl.layer_norm(conv5)

      print 'x:',x
      print 'conv1:',conv1
      print 'conv2:',conv2
      print 'conv3:',conv3
      print 'conv4:',conv4
      print 'conv5:',conv5
      return conv5


