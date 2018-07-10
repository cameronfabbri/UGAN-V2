'''

   Main training file

   The goal is to correct the colors in underwater images.
   CycleGAN was used to create images that appear to be underwater.
   Those will be sent into the generator, which will attempt to correct the
   colors.

'''

import cPickle as pickle
import tensorflow as tf
from scipy import misc
from tqdm import tqdm
import numpy as np
import argparse
import ntpath
import random
import glob
import time
import sys
import cv2
import os

# my imports
sys.path.insert(0, 'ops/')
sys.path.insert(0, 'nets/')
from tf_ops import *
import data_ops

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--LEARNING_RATE', required=False,default=1e-4,type=float,help='Learning rate')
   parser.add_argument('--LOSS_METHOD',   required=False,default='wgan',help='Loss function for GAN')
   parser.add_argument('--BATCH_SIZE',    required=False,default=32,type=int,help='Batch size')
   parser.add_argument('--NUM_LAYERS',    required=False,default=16,type=int,help='Number of total layers in G')
   parser.add_argument('--L1_WEIGHT',     required=False,default=100.,type=float,help='Weight for L1 loss')
   parser.add_argument('--IG_WEIGHT',     required=False,default=0.,type=float,help='Weight for image gradient loss')
   parser.add_argument('--NETWORK',       required=False,default='pix2pix',type=str,help='Network to use')
   parser.add_argument('--AUGMENT',       required=False,default=0,type=int,help='Augment data or not')
   parser.add_argument('--EPOCHS',        required=False,default=100,type=int,help='Number of epochs for GAN')
   parser.add_argument('--DATA',          required=False,default='underwater_imagenet',type=str,help='Dataset to use')
   parser.add_argument('--LAB',           required=False,default=0,type=int,help='LAB colorspace option')
   a = parser.parse_args()

   LEARNING_RATE = float(a.LEARNING_RATE)
   LOSS_METHOD   = a.LOSS_METHOD
   BATCH_SIZE    = a.BATCH_SIZE
   NUM_LAYERS    = a.NUM_LAYERS
   L1_WEIGHT     = float(a.L1_WEIGHT)
   IG_WEIGHT     = float(a.IG_WEIGHT)
   NETWORK       = a.NETWORK
   AUGMENT       = a.AUGMENT
   EPOCHS        = a.EPOCHS
   DATA          = a.DATA
   LAB           = bool(a.LAB)
   
   EXPERIMENT_DIR  = 'checkpoints/LOSS_METHOD_'+LOSS_METHOD\
                     +'/NETWORK_'+NETWORK\
                     +'/NUM_LAYERS_'+str(NUM_LAYERS)\
                     +'/L1_WEIGHT_'+str(L1_WEIGHT)\
                     +'/IG_WEIGHT_'+str(IG_WEIGHT)\
                     +'/AUGMENT_'+str(AUGMENT)\
                     +'/DATA_'+DATA\
                     +'/LAB_'+str(LAB)+'/'\

   IMAGES_DIR      = EXPERIMENT_DIR+'images/'

   print
   print 'Creating',EXPERIMENT_DIR
   try: os.makedirs(IMAGES_DIR)
   except: pass
   try: os.makedirs(TEST_IMAGES_DIR)
   except: pass

   exp_info = dict()
   exp_info['LEARNING_RATE'] = LEARNING_RATE
   exp_info['LOSS_METHOD']   = LOSS_METHOD
   exp_info['BATCH_SIZE']    = BATCH_SIZE
   exp_info['NUM_LAYERS']    = NUM_LAYERS
   exp_info['L1_WEIGHT']     = L1_WEIGHT
   exp_info['IG_WEIGHT']     = IG_WEIGHT
   exp_info['NETWORK']       = NETWORK
   exp_info['AUGMENT']       = AUGMENT
   exp_info['EPOCHS']        = EPOCHS
   exp_info['DATA']          = DATA
   exp_info['LAB']           = LAB
   exp_pkl = open(EXPERIMENT_DIR+'info.pkl', 'wb')
   data = pickle.dumps(exp_info)
   exp_pkl.write(data)
   exp_pkl.close()
   
   print
   print 'LEARNING_RATE: ',LEARNING_RATE
   print 'LOSS_METHOD:   ',LOSS_METHOD
   print 'BATCH_SIZE:    ',BATCH_SIZE
   print 'NUM_LAYERS:    ',NUM_LAYERS
   print 'L1_WEIGHT:     ',L1_WEIGHT
   print 'IG_WEIGHT:     ',IG_WEIGHT
   print 'NETWORK:       ',NETWORK
   print 'AUGMENT:       ',AUGMENT
   print 'EPOCHS:        ',EPOCHS
   print 'DATA:          ',DATA
   print 'LAB:           ',LAB
   print

   if NETWORK == 'pix2pix': from pix2pix import *
   if NETWORK == 'resnet': from resnet import *

   # global step that is saved with a model to keep track of how many steps/epochs
   global_step = tf.Variable(0, name='global_step', trainable=False)

   # underwater image
   image_u = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 256, 256, 3), name='image_u')
   if LAB: image_u = rgb_to_lab(image_u)

   # correct image
   image_r = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 256, 256, 3), name='image_r')
   if LAB: image_r = rgb_to_lab(image_r)

   # generated corrected colors
   if NUM_LAYERS == 16:
      layers     = netG16_encoder(image_u)
      gen_image  = netG16_decoder(layers)
   if NUM_LAYERS == 8:
      layers     = netG8_encoder(image_u)
      gen_image  = netG8_decoder(layers)

   # send 'clean' water images to D
   D_real = netD(image_r, LOSS_METHOD)

   # send corrected underwater images to D
   D_fake = netD(gen_image, LOSS_METHOD, reuse=True)

   e = 1e-12
   if LOSS_METHOD == 'least_squares':
      print 'Using least squares loss'
      errD_real = tf.nn.sigmoid(D_real)
      errD_fake = tf.nn.sigmoid(D_fake)
      errG = 0.5*(tf.reduce_mean(tf.square(errD_fake - 1)))
      errD = tf.reduce_mean(0.5*(tf.square(errD_real - 1)) + 0.5*(tf.square(errD_fake)))
   if LOSS_METHOD == 'gan':
      print 'Using original GAN loss'
      errD_real = tf.nn.sigmoid(D_real)
      errD_fake = tf.nn.sigmoid(D_fake)
      errG = tf.reduce_mean(-tf.log(errD_fake + e))
      errD = tf.reduce_mean(-(tf.log(errD_real+e)+tf.log(1-errD_fake+e)))
   if LOSS_METHOD == 'wgan':
      # cost functions
      errD = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
      errG = -tf.reduce_mean(D_fake)

      # gradient penalty
      epsilon = tf.random_uniform([], 0.0, 1.0)
      x_hat = image_r*epsilon + (1-epsilon)*gen_image
      d_hat = netD(x_hat, LOSS_METHOD, reuse=True)
      gradients = tf.gradients(d_hat, x_hat)[0]
      slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
      gradient_penalty = 10*tf.reduce_mean((slopes-1.0)**2)
      errD += gradient_penalty

   if L1_WEIGHT > 0.0:
      l1_loss = tf.reduce_mean(tf.abs(gen_image-image_r))
      errG += L1_WEIGHT*l1_loss

   if IG_WEIGHT > 0.0:
      ig_loss = loss_gradient_difference(image_r, image_u)
      errG += IG_WEIGHT*ig_loss

   # tensorboard summaries
   tf.summary.scalar('d_loss', tf.reduce_mean(errD))
   tf.summary.scalar('g_loss', tf.reduce_mean(errG))
   try: tf.summary.scalar('l1_loss', tf.reduce_mean(l1_loss))
   except: pass
   try: tf.summary.scalar('ig_loss', tf.reduce_mean(ig_loss))
   except: pass

   # get all trainable variables, and split by network G and network D
   t_vars = tf.trainable_variables()
   d_vars = [var for var in t_vars if 'd_' in var.name]
   g_vars = [var for var in t_vars if 'g_' in var.name]
      
   G_train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(errG, var_list=g_vars, global_step=global_step)
   D_train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(errD, var_list=d_vars)

   saver = tf.train.Saver(max_to_keep=2)

   init = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
   sess = tf.Session()
   sess.run(init)

   summary_writer = tf.summary.FileWriter(EXPERIMENT_DIR+'/logs/', graph=tf.get_default_graph())

   ckpt = tf.train.get_checkpoint_state(EXPERIMENT_DIR)
   if ckpt and ckpt.model_checkpoint_path:
      print "Restoring previous model..."
      try:
         saver.restore(sess, ckpt.model_checkpoint_path)
         print "Model restored"
      except:
         print "Could not restore model"
         pass
   
   step = int(sess.run(global_step))

   merged_summary_op = tf.summary.merge_all()

   # underwater photos
   trainA_paths = np.asarray(glob.glob('datasets/'+DATA+'/trainA/*.jpg'))
   # normal photos (ground truth)
   trainB_paths = np.asarray(glob.glob('datasets/'+DATA+'/trainB/*.jpg'))
   # testing paths
   test_paths = np.asarray(glob.glob('datasets/'+DATA+'/test/*.jpg'))
   #test_paths = np.asarray(glob.glob('datasets/'+DATA+'/trainA/*.jpg'))

   print len(trainB_paths),'training pairs'
   num_train = len(trainB_paths)
   # flip up down, flip left right, send in clean, blur, interpolate
   num_test  = len(test_paths)

   n_critic = 1
   if LOSS_METHOD == 'wgan': n_critic = 5

   epoch_num = step/(num_train*2*2*2*2*5/BATCH_SIZE)


   while epoch_num < EPOCHS:
      s = time.time()
      epoch_num = step/(num_train*2*2*2*2*5/BATCH_SIZE)

      # pick random images every time for D
      for itr in xrange(n_critic):
         idx = np.random.choice(np.arange(num_train), BATCH_SIZE, replace=False)
         batchA_paths = trainA_paths[idx]
         batchB_paths = trainB_paths[idx]
         batchA_images = np.empty((BATCH_SIZE, 256, 256, 3), dtype=np.float32)
         batchB_images = np.empty((BATCH_SIZE, 256, 256, 3), dtype=np.float32)
         i = 0
         for a,b in zip(batchA_paths, batchB_paths):
            a_img = misc.imread(a)
            b_img = misc.imread(b)
            # Data augmentation here - each has 50% chance
            if AUGMENT: a_img, b_img = data_ops.augment(a_img, b_img)
            batchA_images[i, ...] = data_ops.preprocess(a_img)
            batchB_images[i, ...] = data_ops.preprocess(b_img)
            i += 1
         sess.run(D_train_op, feed_dict={image_u:batchA_images, image_r:batchB_images})

      # also get new batch for G
      idx = np.random.choice(np.arange(num_train), BATCH_SIZE, replace=False)
      batchA_paths = trainA_paths[idx]
      batchB_paths = trainB_paths[idx]
      batchA_images = np.empty((BATCH_SIZE, 256, 256, 3), dtype=np.float32)
      batchB_images = np.empty((BATCH_SIZE, 256, 256, 3), dtype=np.float32)
      i = 0
      for a,b in zip(batchA_paths, batchB_paths):
         a_img = misc.imread(a)
         b_img = misc.imread(b)
         # Data augmentation here - each has 50% chance
         if AUGMENT: a_img, b_img = data_ops.augment(a_img, b_img)
         batchA_images[i, ...] = data_ops.preprocess(a_img)
         batchB_images[i, ...] = data_ops.preprocess(b_img)
         i += 1
      sess.run(G_train_op, feed_dict={image_u:batchA_images, image_r:batchB_images})

      D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op], feed_dict={image_u:batchA_images, image_r:batchB_images})

      summary_writer.add_summary(summary, step)
      ss = time.time()-s
      print 'epoch:',epoch_num,'step:',step,'D loss:',D_loss,'G_loss:',G_loss,'time:',ss
      step += 1
      
      if step%500 == 0:
         print 'Saving model...'
         saver.save(sess, EXPERIMENT_DIR+'checkpoint-'+str(step))
         saver.export_meta_graph(EXPERIMENT_DIR+'checkpoint-'+str(step)+'.meta')
         print 'Model saved\n'

         idx = np.random.choice(np.arange(num_test), BATCH_SIZE, replace=False)
         batch_paths = test_paths[idx]
         
         batch_images = np.empty((BATCH_SIZE, 256, 256, 3), dtype=np.float32)

         print 'Testing...'
         i = 0
         for a in batch_paths:
            a_img = misc.imread(a).astype('float32')
            a_img = data_ops.preprocess(misc.imresize(a_img, (256, 256, 3)))
            batch_images[i, ...] = a_img
            i += 1

         gen_images = np.asarray(sess.run(gen_image, feed_dict={image_u:batch_images}))

         c = 0
         for gen, real in zip(gen_images, batch_images):
            misc.imsave(IMAGES_DIR+str(step)+'_real.png', real)
            misc.imsave(IMAGES_DIR+str(step)+'_gen.png', gen)
            c += 1
            if c == 5: break
         print 'Done with test images'
