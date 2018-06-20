import tensorflow as tf
import tensorflow.contrib.layers as tcl
import sys

sys.path.insert(0, 'ops/')
from tf_ops import *

def netG_encoder(enc, NUM_LAYERS):

   print 'input:',enc

   layers = []

   # feature maps
   fm = 64

   for l in range(NUM_LAYERS/2):
      enc = tcl.conv2d(enc, fm, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_enc_'+str(l))
      enc = tcl.batch_norm(enc)
      enc = lrelu(enc)
      fm = fm*2
      if fm > 512: fm = 512
      layers.append(enc)

   for l in layers:
      print 'enc:',l
   
   return layers

def netG_decoder(layers, NUM_LAYERS):
   print
   # get feature maps
   fm = layers[-1].get_shape().as_list()[-1]

   dec = layers[-1]

   '''
      This overly confusing for loop is able to account for dynamic amounts of layers
      with skip connections.
   '''
   for l in range(NUM_LAYERS/2):
      skip_layer = NUM_LAYERS/2 - l - 2
      if skip_layer == -1: break
      if dec.get_shape().as_list()[1] < 16: fm = 1024
      dec = tcl.convolution2d_transpose(dec, fm/2, 4, 2, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_dec_'+str(l))
      #print 'concat',dec,'with',layers[skip_layer]
      dec = tf.concat([dec, layers[skip_layer]], axis=3)
      fm = fm/2
      if dec.get_shape().as_list()[1] == 256: break
      print 'dec:',dec
   dec = tcl.convolution2d_transpose(dec, 3, 4, 2, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_dec_out')
   print 'output:',dec
   return dec

def netD(x, LOSS_METHOD, reuse=False):
   print
   print 'netD'

   sc = tf.get_variable_scope()
   with tf.variable_scope(sc, reuse=reuse):

      conv1 = tcl.conv2d(x, 64, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv1')
      if LOSS_METHOD != 'wgan': conv1 = tcl.batch_norm(conv1)
      conv1 = lrelu(conv1)
      
      conv2 = tcl.conv2d(conv1, 128, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv2')
      if LOSS_METHOD != 'wgan': conv2 = tcl.batch_norm(conv2)
      conv2 = lrelu(conv2)
      
      conv3 = tcl.conv2d(conv2, 256, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv3')
      if LOSS_METHOD != 'wgan': conv3 = tcl.batch_norm(conv3)
      conv3 = lrelu(conv3)
      
      conv4 = tcl.conv2d(conv3, 512, 4, 1, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv4')
      if LOSS_METHOD != 'wgan': conv4 = tcl.batch_norm(conv4)
      conv4 = lrelu(conv4)
      
      conv5 = tcl.conv2d(conv4, 1, 1, 1, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv5')
      if LOSS_METHOD != 'wgan': conv5 = tcl.batch_norm(conv5)

      print 'x:',x
      print 'conv1:',conv1
      print 'conv2:',conv2
      print 'conv3:',conv3
      print 'conv4:',conv4
      print 'conv5:',conv5
      return conv5


