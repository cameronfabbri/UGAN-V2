import tensorflow as tf
import tensorflow.contrib.layers as tcl
import sys

sys.path.insert(0, 'ops/')
from tf_ops import *

def encoder(x):
      
   enc_conv1 = tcl.conv2d(x, 16, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_enc_conv1')
   enc_conv1 = tcl.batch_norm(enc_conv1)
   enc_conv1 = lrelu(enc_conv1)
   
   enc_conv2 = tcl.conv2d(enc_conv1, 32, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_enc_conv2')
   enc_conv2 = tcl.batch_norm(enc_conv2)
   enc_conv2 = lrelu(enc_conv2)
   
   enc_conv3 = tcl.conv2d(enc_conv2, 64, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_enc_conv3')
   enc_conv3 = tcl.batch_norm(enc_conv3)
   enc_conv3 = lrelu(enc_conv3)

   enc_conv4 = tcl.conv2d(enc_conv3, 128, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_enc_conv4')
   enc_conv4 = tcl.batch_norm(enc_conv4)
   enc_conv4 = lrelu(enc_conv4)
   
   enc_conv5 = tcl.conv2d(enc_conv4, 256, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_enc_conv5')
   enc_conv5 = tcl.batch_norm(enc_conv5)
   enc_conv5 = lrelu(enc_conv5)

   enc_conv6 = tcl.conv2d(enc_conv5, 512, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_enc_conv6')
   enc_conv6 = tcl.batch_norm(enc_conv6)
   enc_conv6 = lrelu(enc_conv6)
   
   enc_conv7 = tcl.conv2d(enc_conv6, 512, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_enc_conv7')
   enc_conv7 = tcl.batch_norm(enc_conv7)
   enc_conv7 = lrelu(enc_conv7)

   print
   print 'encoder:'
   print 'x:        ',x
   print 'enc_conv1:',enc_conv1
   print 'enc_conv2:',enc_conv2
   print 'enc_conv3:',enc_conv3
   print 'enc_conv4:',enc_conv4
   print 'enc_conv5:',enc_conv5
   print 'enc_conv6:',enc_conv6
   print 'enc_conv7:',enc_conv7
   print
   print
   
   enc = {'x':x,
        'enc_conv1':enc_conv1,
        'enc_conv2':enc_conv2,
        'enc_conv3':enc_conv3,
        'enc_conv4':enc_conv4,
        'enc_conv5':enc_conv5,
        'enc_conv6':enc_conv6,
        'embedding':enc_conv7
   }

   return enc

def decoder(enc, y, skip_connections, upsample):
   print 'decoder:'
   enc_conv1 = enc['enc_conv1']
   enc_conv2 = enc['enc_conv2']
   enc_conv3 = enc['enc_conv3']
   enc_conv4 = enc['enc_conv4']
   enc_conv5 = enc['enc_conv5']
   enc_conv6 = enc['enc_conv6']
   enc_conv7 = enc['embedding']

   # reshape so it's batchx1x1xy_size
   y_dim = int(y.get_shape().as_list()[-1])
   y = tf.reshape(y, shape=[-1, 1, 1, y_dim])

   # concat y onto every layer
   enc_conv7 = conv_cond_concat(enc_conv7, y)

   if upsample == 'transpose_conv':
      dec_conv1 = tcl.convolution2d_transpose(enc_conv7, 512, 4, 1, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_dec_conv1')
   if upsample == 'pixel_shuffle': print 'not implemented yet'
   if upsample == 'upconv': print 'not implemented yet'
   dec_conv1 = relu(dec_conv1)
   if skip_connections: dec_conv1 = tf.concat([dec_conv1, enc_conv7], axis=3)
   dec_conv1 = conv_cond_concat(dec_conv1, y)
   print 'dec_conv1:',dec_conv1

   if upsample == 'transpose_conv':
      dec_conv2 = tcl.convolution2d_transpose(dec_conv1, 512, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_dec_conv2')
   if upsample == 'pixel_shuffle': print 'not implemented yet'
   if upsample == 'upconv': print 'not implemented yet'
   dec_conv2 = relu(dec_conv2)
   if skip_connections: dec_conv2 = tf.concat([dec_conv2, enc_conv6], axis=3)
   dec_conv2 = conv_cond_concat(dec_conv2, y)
   print 'dec_conv2:',dec_conv2

   if upsample == 'transpose_conv':
      dec_conv3 = tcl.convolution2d_transpose(dec_conv2, 256, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_dec_conv3')
   if upsample == 'pixel_shuffle': print 'not implemented yet'
   if upsample == 'upconv': print 'not implemented yet'
   dec_conv3 = relu(dec_conv3)
   if skip_connections: dec_conv3 = tf.concat([dec_conv3, enc_conv5], axis=3)
   dec_conv3 = conv_cond_concat(dec_conv3, y)
   print 'dec_conv3:',dec_conv3

   if upsample == 'transpose_conv':
      dec_conv4 = tcl.convolution2d_transpose(dec_conv3, 128, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_dec_conv4')
   if upsample == 'pixel_shuffle': print 'not implemented yet'
   if upsample == 'upconv': print 'not implemented yet'
   dec_conv4 = relu(dec_conv4)
   if skip_connections: dec_conv4 = tf.concat([dec_conv4, enc_conv4], axis=3)
   dec_conv4 = conv_cond_concat(dec_conv4, y)
   print 'dec_conv4:',dec_conv4
   
   if upsample == 'transpose_conv':
      dec_conv5 = tcl.convolution2d_transpose(dec_conv4, 64, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_dec_conv5')
   if upsample == 'pixel_shuffle': print 'not implemented yet'
   if upsample == 'upconv': print 'not implemented yet'
   dec_conv5 = relu(dec_conv5)
   if skip_connections: dec_conv5 = tf.concat([dec_conv5, enc_conv3], axis=3)
   dec_conv5 = conv_cond_concat(dec_conv5, y)
   print 'dec_conv5:',dec_conv5

   if upsample == 'transpose_conv':
      dec_conv6 = tcl.convolution2d_transpose(dec_conv5, 32, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_dec_conv6')
   if upsample == 'pixel_shuffle': print 'not implemented yet'
   if upsample == 'upconv': print 'not implemented yet'
   dec_conv6 = relu(dec_conv6)
   if skip_connections: dec_conv6 = tf.concat([dec_conv6, enc_conv2], axis=3)
   dec_conv6 = conv_cond_concat(dec_conv6, y)
   print 'dec_conv6:',dec_conv6

   if upsample == 'transpose_conv':
      dec_conv7 = tcl.convolution2d_transpose(dec_conv6, 16, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_dec_conv7')
   if upsample == 'pixel_shuffle': print 'not implemented yet'
   if upsample == 'upconv': print 'not implemented yet'
   dec_conv7 = relu(dec_conv7)
   if skip_connections: dec_conv7 = tf.concat([dec_conv7, enc_conv1], axis=3)
   dec_conv7 = conv_cond_concat(dec_conv7, y)
   print 'dec_conv7:',dec_conv7

   if upsample == 'transpose_conv':
      dec_conv8 = tcl.convolution2d_transpose(dec_conv7, 3, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_dec_conv8')
   if upsample == 'pixel_shuffle': print 'not implemented yet'
   if upsample == 'upconv': print 'not implemented yet'
   dec_conv8 = tanh(dec_conv8)
   print 'dec_conv8', dec_conv8
   print
   return dec_conv8


def netD(embedding, reuse=False):

   #sc = tf.get_variable_scope()
   #with tf.variable_scope(sc, reuse=reuse):

   conv1 = tcl.conv2d(embedding, 512, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv1')
   conv1 = tcl.batch_norm(conv1)
   conv1 = lrelu(conv1)

   conv1 = tcl.flatten(conv1)

   fc1 = tcl.fully_connected(conv1, 512, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_fc1')
   fc1 = lrelu(fc1)
   
   fc2 = tcl.fully_connected(fc1, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_fc2')
   #fc2 = sig(fc2)

   print 'conv1:',conv1
   print 'fc1:',fc1
   print 'fc2:',fc2

   return fc2
