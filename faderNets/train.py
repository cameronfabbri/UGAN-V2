'''

   Main training file for Fader Networks

'''
import scipy.misc as misc
import cPickle as pickle
import tensorflow as tf
import numpy as np
import argparse
import random
import ntpath
import time
import glob
import sys
import os

sys.path.insert(0, 'ops/')
sys.path.insert(0, 'nets/')

import data_ops
import tf_ops

if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument('--batch_size', required=False, type=int, default=32,
                        help='Batch size to use')
   parser.add_argument('--data_dir', required=True, type=str,
                        help='Directory where data is')
   parser.add_argument('--epochs',required=False, type=int, default=500,
                        help='Maximum training epochs')
   parser.add_argument('--ae_loss', required=False, type=str, default='l2',
                        help='Loss for the autoencoder', choices=['l2', 'gan', 'wgan'])
   parser.add_argument('--l1_weight', required=False, type=float, default=0.0,
                        help='l1 weight. Primarily used when using a GAN for the ae loss')
   parser.add_argument('--instance_norm',required=False, type=str, default=0,
                        help='Instance norm instead of batch norm')
   parser.add_argument('--skip_connections',required=False, type=int, default=0,
                        help='Skip connection use in the ae')
   parser.add_argument('--decoder_dropout',required=False, type=float, default=0.,
                        help='Dropout in the decoder')
   parser.add_argument('--lat_dis_dropout',required=False, type=float, default=0.3,
                        help='Dropout in the latent discriminator')
   parser.add_argument('--smooth_label',required=False, type=float, default=0.2,
                        help='Smooth the label for the patch discriminator')
   parser.add_argument('--lambda_ae',required=False, type=float, default=1.,
                        help='Autoencoder loss coefficient')
   parser.add_argument('--lambda_lat_dis',required=False, type=float, default=0.0001,
                        help='Latent discriminator loss feedback coefficient')
   parser.add_argument('--lambda_clf_dis',required=False, type=float, default=0.,
                        help='Classifier discriminator loss feedback coefficient')
   parser.add_argument('--lambda_schedule',required=False, type=int, default=500000,
                        help='Progressively increase discriminators lambda (0 to disable)')
   parser.add_argument('--upsample', required=False, type=str, default='transpose_conv',
                        help='Type of upsampling to use in the decoder',
                        choices=['transpose_conv', 'pixel_shuffle', 'upconv'])
   parser.add_argument('--epoch_size',required=False, type=int, default=50000,
                        help='Size of the epoch (0 to use number of training samples)')
   parser.add_argument('--network',required=False, type=str, default='pix2pix',
                        help='Network architecture to use', choices=['pix2pix', 'resnet'])
   a = parser.parse_args()

   batch_size       = a.batch_size
   data_dir         = a.data_dir
   epochs           = a.epochs
   ae_loss          = a.ae_loss
   l1_weight        = a.l1_weight
   instance_norm    = bool(a.instance_norm)
   skip_connections = bool(a.skip_connections)
   decoder_dropout  = a.decoder_dropout
   lat_dis_dropout  = a.lat_dis_dropout
   smooth_label     = a.smooth_label
   lambda_ae        = a.lambda_ae
   lambda_lat_dis   = a.lambda_lat_dis
   lambda_clf_dis   = a.lambda_clf_dis
   lambda_schedule  = a.lambda_schedule
   upsample         = a.upsample
   epoch_size       = a.epoch_size
   network          = a.network

   CHECKPOINT_DIR = 'checkpoints/ae_loss_'+ae_loss\
                   +'/l1_weight_'+str(l1_weight)\
                   +'/instance_norm_'+str(instance_norm)\
                   +'/skip_connections'+str(skip_connections)\
                   +'/decoder_dropout_'+str(decoder_dropout)\
                   +'/lat_dis_dropout_'+str(lat_dis_dropout)\
                   +'/smooth_label_'+str(smooth_label)\
                   +'/lambda_ae_'+str(lambda_ae)\
                   +'/lambda_lat_dis_'+str(lambda_lat_dis)\
                   +'/lambda_clf_dis_'+str(lambda_clf_dis)\
                   +'/lambda_schedule_'+str(lambda_schedule)\
                   +'/upsample_'+upsample+'/'\

   IMAGES_DIR     = CHECKPOINT_DIR+'images/'

   try: os.makedirs(IMAGES_DIR)
   except: pass

   exp_info = dict()
   exp_info['checkpoint_dir']   = CHECKPOINT_DIR
   exp_info['batch_size']       = batch_size
   exp_info['data_dir']         = data_dir
   exp_info['epochs']           = epochs
   exp_info['ae_loss']          = ae_loss
   exp_info['l1_weight']        = l1_weight
   exp_info['instance_norm']    = instance_norm
   exp_info['skip_connections'] = skip_connections
   exp_info['decoder_dropout']  = decoder_dropout
   exp_info['lat_dis_dropout']  = lat_dis_dropout
   exp_info['smooth_label']     = smooth_label
   exp_info['lambda_ae']        = lambda_ae
   exp_info['lambda_lat_dis']   = lambda_lat_dis
   exp_info['lambda_clf_dis']   = lambda_clf_dis
   exp_info['lambda_schedule']  = lambda_schedule
   exp_info['upsample']         = upsample
   exp_info['epoch_size']       = epoch_size
   exp_info['network']          = network

   exp_pkl = open(CHECKPOINT_DIR+'info.pkl', 'wb')
   data = pickle.dumps(exp_info)
   exp_pkl.write(data)
   exp_pkl.close()

   # import the correct network we are using
   if network == 'pix2pix': from pix2pix import *
   if network == 'resnet': from resnet import *
   
   # placeholders for data going into the network
   global_step = tf.Variable(0, name='global_step', trainable=False)
   x           = tf.placeholder(tf.float32, shape=(batch_size, 256, 256, 3), name='real_images')
   y           = tf.placeholder(tf.float32, shape=(batch_size, 2), name='y')

   # lambda on the discriminator. Start at 0, and increase to lambda_lat_dis
   d_lambda = tf.placeholder(tf.float32, name='d_lambda')

   # the embedding of the image - this is a dictionary of all layers in case we use skip connections
   enc = encoder(x)
   embedding = enc['embedding']

   decoded = decoder(enc, y, skip_connections, upsample)
   
   # D's prediction on which class the embedding is
   logitsD = tf.nn.softmax(netD(embedding))

   # discriminator loss - minimize cross entropy of predicting the label
   errD = tf.multiply(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logitsD)), d_lambda)

   # add the discriminator loss to the generator
   errG = -errD

   # generator loss
   errG = errG + tf.reduce_mean(tf.losses.mean_squared_error(x,decoded,weights=lambda_ae))

   # tensorboard summaries
   tf.summary.scalar('d_loss', errD)
   tf.summary.scalar('g_loss', errG)
   merged_summary_op = tf.summary.merge_all()

   # get all trainable variables, and split by network G and network D
   t_vars = tf.trainable_variables()
   d_vars = [var for var in t_vars if 'd_' in var.name]
   g_vars = [var for var in t_vars if 'g_' in var.name]

   #train_op = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(errG+errD, global_step=global_step)
   G_train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(errG, var_list=g_vars, global_step=global_step)
   D_train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(errD, var_list=d_vars)

   saver = tf.train.Saver(max_to_keep=1)
   init  = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
   sess  = tf.Session()
   sess.run(init)

   # write losses to tf summary to view in tensorboard
   try: tf.summary.scalar('d_loss', tf.reduce_mean(errD))
   except:pass
   try: tf.summary.scalar('g_loss', tf.reduce_mean(errG))
   except:pass

   summary_writer = tf.summary.FileWriter(CHECKPOINT_DIR+'/'+'logs/', graph=tf.get_default_graph())

   # restore previous model if there is one
   ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
   if ckpt and ckpt.model_checkpoint_path:
      print 'Restoring previous model...'
      try:
         saver.restore(sess, ckpt.model_checkpoint_path)
         print 'Model restored'
      except:
         print 'Could not restore model'
         pass
   
   ########################################### training portion

   step = sess.run(global_step)

   train_distorted_paths    = np.asarray(glob.glob('/mnt/data1/images/ugan_dataset/distorted/train/*.JPEG'))[:16]
   train_nondistorted_paths = np.asarray(glob.glob('/mnt/data1/images/ugan_dataset/nondistorted/train/*.JPEG'))[:16]
   
   test_distorted_paths    = np.asarray(glob.glob('/mnt/data1/images/ugan_dataset/distorted/test/*.JPEG'))[:16]
   test_nondistorted_paths = np.asarray(glob.glob('/mnt/data1/images/ugan_dataset/nondistorted/test/*.JPEG'))[:16]

   dlen  = len(train_distorted_paths)
   ndlen = len(train_nondistorted_paths)

   t_dlen  = len(test_distorted_paths)
   t_ndlen = len(test_nondistorted_paths)
   
   # create labels: d: [0, 1], nd: [1, 0]
   dlabels_  = [0, 1]
   dlabels   = np.asarray([dlabels_]*dlen)
   ndlabels_ = [1, 0]
   ndlabels  = np.asarray([ndlabels_]*ndlen)

   # test labels (same thing, but could have different lengths)
   t_dlabels  = np.asarray([dlabels_]*t_dlen)
   t_ndlabels = np.asarray([ndlabels_]*t_ndlen)

   # find the d_lambda
   alpha = np.linspace(0,1, num=500000)
   d_lambdas = []
   x1 = 0
   x2 = lambda_lat_dis
   for a in alpha:
      l = x1*(1-a) + x2*a
      d_lambdas.append(l)

   while True:

      if step > 499999: step_ = lambda_lat_dis
      else: step_ = step

      # put in batch of distorted first, then non_distorted
      idx          = np.random.choice(np.arange(dlen), batch_size, replace=False)
      batch_y      = dlabels[idx]
      batch_paths  = train_distorted_paths[idx]
      batch_images = np.empty((batch_size, 256, 256, 3), dtype=np.float32)
      i = 0
      for img in batch_paths:
         img = misc.imread(img)
         img = misc.imresize(img, (256,256))
         img = data_ops.normalize(img)
         batch_images[i, ...] = img
         i += 1
      
      #sess.run(train_op, feed_dict={x:batch_images, y:batch_y, d_lambda:d_lambdas[step_]})
      sess.run(D_train_op, feed_dict={x:batch_images, y:batch_y, d_lambda:d_lambdas[step_]})
      sess.run(G_train_op, feed_dict={x:batch_images, y:batch_y, d_lambda:d_lambdas[step_]})
      
      idx          = np.random.choice(np.arange(ndlen), batch_size, replace=False)
      batch_y      = ndlabels[idx]
      batch_paths  = train_nondistorted_paths[idx]
     
      batch_images = np.empty((batch_size, 256, 256, 3), dtype=np.float32)
      i = 0
      for img in batch_paths:
         img = misc.imread(img)
         img = misc.imresize(img, (256,256))
         img = data_ops.normalize(img)
         batch_images[i, ...] = img
         i += 1
     
      #sess.run(train_op, feed_dict={x:batch_images, y:batch_y, d_lambda:d_lambdas[step_]})
      sess.run(D_train_op, feed_dict={x:batch_images, y:batch_y, d_lambda:d_lambdas[step_]})
      sess.run(G_train_op, feed_dict={x:batch_images, y:batch_y, d_lambda:d_lambdas[step_]})
      
      D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op],
                                 feed_dict={x:batch_images, y:batch_y, d_lambda:d_lambdas[step_]})

      summary_writer.add_summary(summary, step)

      print 'step:',step,'D loss:',D_loss,'G_loss:',G_loss
      step += 1

      '''
         Save model and run tests. Save out some going from distorted -> nondistorted, and other way around
      '''
      if step%500 == 0:
         print 'Saving model...'
         saver.save(sess, CHECKPOINT_DIR+'checkpoint-'+str(step))
         saver.export_meta_graph(CHECKPOINT_DIR+'checkpoint-'+str(step)+'.meta')

         # distorted -> non distorted
         idx          = np.random.choice(np.arange(t_dlen), batch_size, replace=False)
         batch_y      = t_ndlabels[idx]
         batch_paths  = test_distorted_paths[idx]
         batch_images1 = np.empty((batch_size, 256, 256, 3), dtype=np.float32)
         i = 0
         for img in batch_paths:
            img = misc.imread(img)
            img = misc.imresize(img, (256,256))
            img = data_ops.normalize(img)
            batch_images1[i, ...] = img
            i += 1
         dec1 = np.asarray(sess.run(decoded, feed_dict={x:batch_images1, y:batch_y, d_lambda:d_lambdas[step_]}))

         # non distorted -> distorted
         idx          = np.random.choice(np.arange(t_ndlen), batch_size, replace=False)
         batch_y      = t_dlabels[idx]
         batch_paths  = test_nondistorted_paths[idx]
         batch_images2 = np.empty((batch_size, 256, 256, 3), dtype=np.float32)
         i = 0
         for img in batch_paths:
            img = misc.imread(img)
            img = misc.imresize(img, (256,256))
            img = data_ops.normalize(img)
            batch_images2[i, ...] = img
            i += 1
         dec2 = np.asarray(sess.run(decoded, feed_dict={x:batch_images2, y:batch_y, d_lambda:d_lambdas[step_]}))

         num = 0
         for img1,img2 in zip(dec1, batch_images1):
            blank = np.zeros((256*2, 256, 3))
            img1 = (img1+1.)
            img1 *= 127.5
            img1 = np.clip(img1, 0, 255).astype(np.uint8)
            
            img2 = (img2+1.)
            img2 *= 127.5
            img2 = np.clip(img2, 0, 255).astype(np.uint8)

            img = np.concatenate((img2, img1), axis=1)

            misc.imsave(IMAGES_DIR+'step_'+str(step)+'AB_num_'+str(num)+'.png', img)
            num += 1
            if num == 3: break
         
         num = 0
         for img1,img2 in zip(dec2, batch_images2):
            blank = np.zeros((256*2, 256, 3))
            img1 = (img1+1.)
            img1 *= 127.5
            img1 = np.clip(img1, 0, 255).astype(np.uint8)
            
            img2 = (img2+1.)
            img2 *= 127.5
            img2 = np.clip(img2, 0, 255).astype(np.uint8)

            img = np.concatenate((img2, img1), axis=1)

            misc.imsave(IMAGES_DIR+'step_'+str(step)+'BA_num_'+str(num)+'.png', img)
            num += 1
            if num == 3: break
   
   saver.save(sess, CHECKPOINT_DIR+'checkpoint-'+str(step))
   saver.export_meta_graph(CHECKPOINT_DIR+'checkpoint-'+str(step)+'.meta')
