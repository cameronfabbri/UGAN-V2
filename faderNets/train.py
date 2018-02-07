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
   parser.add_argument('--batch_size', required=False, type=int, default=64,
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

   # the embedding of the image - this is a dictionary of all layers
   enc = encoder(x)
   embedding = enc['embedding']

   decoded = decoder(enc, y, skip_connections, upsample)

   # D's prediction on which class the embedding is
   logitsD = tf.nn.softmax(netD(embedding))

   # loss on D - cross entropy with real class y
   errD = -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logitsD))

   errG = tf.reduce_mean(tf.nn.l2_loss(x-decoded))

   # tensorboard summaries
   tf.summary.scalar('d_loss', errD)
   tf.summary.scalar('g_loss', errG)
   merged_summary_op = tf.summary.merge_all()

   # get all trainable variables, and split by network G and network D
   t_vars = tf.trainable_variables()
   d_vars = [var for var in t_vars if 'd_' in var.name]
   g_vars = [var for var in t_vars if 'g_' in var.name]

   train_op = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(errG+errD)

   # optimize G
   #G_train_op = tf.train.AdamOptimizer(learning_rate=lr,beta1=beta1,beta2=beta2).minimize(errG, var_list=g_vars, global_step=global_step)
   # optimize D
   #D_train_op = tf.train.AdamOptimizer(learning_rate=lr,beta1=beta1,beta2=beta2).minimize(errD, var_list=d_vars)

   saver = tf.train.Saver(max_to_keep=1)
   init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
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

   train_distorted_paths    = np.asarray(glob.glob('/mnt/data1/images/ugan_dataset/distorted/*.JPEG'))
   train_nondistorted_paths = np.asarray(glob.glob('/mnt/data1/images/ugan_dataset/nondistorted/*.JPEG'))

   dlen  = len(train_distorted_paths)
   ndlen = len(train_nondistorted_paths)

   # create labels: d: [0, 1], nd: [1, 0]
   dlabels = [0, 1]
   dlabels = np.asarray([dlabels]*dlen)
   
   ndlabels = [1, 0]
   ndlabels = np.asarray([ndlabels]*ndlen)

   while True:

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
      
      sess.run(train_op, feed_dict={x:batch_images, y:batch_y})
      
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
     
      sess.run(train_op, feed_dict={x:batch_images, y:batch_y})
      
      D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op],
                                 feed_dict={x:batch_images, y:batch_y})

      summary_writer.add_summary(summary, step)

      print 'step:',step,'D loss:',D_loss,'G_loss:',G_loss
      step += 1


      if step%500 == 0:
         print 'Saving model...'
         saver.save(sess, CHECKPOINT_DIR+'checkpoint-'+str(step))
         saver.export_meta_graph(CHECKPOINT_DIR+'checkpoint-'+str(step)+'.meta')

         idx          = np.random.choice(np.arange(test_len), batch_size, replace=False)
         batch_z      = np.random.normal(0.0, 1.0, size=[batch_size, 100]).astype(np.float32)
         batch_y      = test_annots[idx]
         batch_images = test_images[idx]

         gen_imgs = np.squeeze(np.asarray(sess.run([gen_images],
                                 feed_dict={z:batch_z, y:batch_y, real_images:batch_images})))
         num = 0
         for img,atr in zip(gen_imgs, batch_y):
            img = (img+1.)
            img *= 127.5
            img = np.clip(img, 0, 255).astype(np.uint8)
            img = np.reshape(img, (64, 64, -1))
            misc.imsave(IMAGES_DIR+'step_'+str(step)+'_num_'+str(num)+'.png', img)
            with open(IMAGES_DIR+'attrs.txt', 'a') as f:
               f.write('step_'+str(step)+'_num_'+str(num)+','+str(atr)+'\n')
            num += 1
            if num == 5: break
   
   saver.save(sess, CHECKPOINT_DIR+'checkpoint-'+str(step))
   saver.export_meta_graph(CHECKPOINT_DIR+'checkpoint-'+str(step)+'.meta')
