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
from skimage import io, color, transform
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
sys.path.insert(0, 'measures/')
sys.path.insert(0, 'nets/')
from tf_ops import *
import data_ops
import uiqm

'''
   don't need to load underwater ab channels
   also since we have image pairs, only need to load up one image L channel
'''
def loadImages(DATA):
   trainA_paths  = data_ops.getPaths('datasets/'+DATA+'/trainA/') # underwater photos
   trainB_paths  = data_ops.getPaths('datasets/'+DATA+'/trainB/') # normal photos (ground truth)
   train_L  = np.empty((len(trainA_paths), 256, 256, 1), dtype=np.float64)
   train_ab = np.empty((len(trainB_paths), 256, 256, 2), dtype=np.float64)
   j = 0
   for a,b in tqdm(zip(trainA_paths, trainB_paths)):
      b_img = io.imread(b)
      b_img = color.rgb2lab(b_img)
      img_L = b_img[:,:,0]
      img_L = np.expand_dims(img_L, 2)
      img_L = img_L/50.0-1.
      b_img_ab = b_img[:,:,1:]
      b_img_ab = b_img_ab/128.
      train_L[j, ...]  = img_L
      train_ab[j,...] = b_img_ab
      j += 1
   return train_L, train_ab


if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--LEARNING_RATE', required=False,default=1e-4,type=float,help='Learning rate')
   parser.add_argument('--LOSS_METHOD',   required=False,default='wgan',help='Loss function for GAN')
   parser.add_argument('--UIQM_WEIGHT',   required=False,default=0.0,type=float,help='UIQM loss weight')
   parser.add_argument('--BATCH_SIZE',    required=False,default=32,type=int,help='Batch size')
   parser.add_argument('--NUM_LAYERS',    required=False,default=16,type=int,help='Number of total layers in G')
   parser.add_argument('--L1_WEIGHT',     required=False,default=100.,type=float,help='Weight for L1 loss')
   parser.add_argument('--IG_WEIGHT',     required=False,default=0.,type=float,help='Weight for image gradient loss')
   parser.add_argument('--NETWORK',       required=False,default='pix2pix',type=str,help='Network to use')
   parser.add_argument('--AUGMENT',       required=False,default=0,type=int,help='Augment data or not')
   parser.add_argument('--EPOCHS',        required=False,default=100,type=int,help='Number of epochs for GAN')
   parser.add_argument('--DATA',          required=False,default='large',type=str,help='Dataset to use')
   parser.add_argument('--LAB',           required=False,default=1,type=int,help='LAB colorspace option')
   a = parser.parse_args()

   LEARNING_RATE = float(a.LEARNING_RATE)
   UIQM_WEIGHT   = a.UIQM_WEIGHT
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
                     +'/UIQM_WEIGHT_'+str(UIQM_WEIGHT)\
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
   exp_info['UIQM_WEIGHT']   = UIQM_WEIGHT
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

   # image L channel
   image_L = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 256, 256, 1), name='image_l')

   # correct ab
   image_ab = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 256, 256, 2), name='image_ab')

   # generated corrected colors
   if NUM_LAYERS == 16:
      layers  = netG16_encoder(image_L)
      gen_ab  = netG16_decoder(layers, lab=True)
   if NUM_LAYERS == 8:
      layers = netG8_encoder(image_L)
      gen_ab = netG8_decoder(layers, lab=True)

   image_Lab = tf.concat([image_L, image_ab], axis=3)
   image_g = tf.concat([image_L, gen_ab], axis=3)

   # send 'clean' water images to D
   D_real = netD(image_Lab, LOSS_METHOD)

   # send corrected underwater images to D
   D_fake = netD(image_g, LOSS_METHOD, reuse=True)

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
      x_hat = image_Lab*epsilon + (1-epsilon)*image_g
      d_hat = netD(x_hat, LOSS_METHOD, reuse=True)
      gradients = tf.gradients(d_hat, x_hat)[0]
      slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
      gradient_penalty = 10*tf.reduce_mean((slopes-1.0)**2)
      errD += gradient_penalty

   if L1_WEIGHT > 0.0:
      l1_loss = tf.reduce_mean(tf.abs(gen_ab-image_ab))
      errG += L1_WEIGHT*l1_loss

   if IG_WEIGHT > 0.0:
      ig_loss = loss_gradient_difference(image_ab, image_L)
      errG += IG_WEIGHT*ig_loss

   if UIQM_WEIGHT > 0.0:
      uiqm_val  = tf.placeholder(tf.float32, shape=(), name='uiqm_val')
      # this is - because we want to maximize the uiqm, so minimize the negative of it (subtract it)
      uiqm_loss = -UIQM_WEIGHT*uiqm_val
      errG += uiqm_loss

   # tensorboard summaries
   tf.summary.scalar('d_loss', tf.reduce_mean(errD))
   tf.summary.scalar('g_loss', tf.reduce_mean(errG))
   try: tf.summary.scalar('l1_loss', tf.reduce_mean(l1_loss))
   except: pass
   try: tf.summary.scalar('ig_loss', tf.reduce_mean(ig_loss))
   except: pass
   try: tf.summary.scalar('uiqm_loss', tf.reduce_mean(uiqm_loss))
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

   trainA_paths = data_ops.getPaths('datasets/'+DATA+'/trainA/') # underwater photos
   trainB_paths = data_ops.getPaths('datasets/'+DATA+'/trainB/') # normal photos (ground truth)
   test_paths   = data_ops.getPaths('datasets/'+DATA+'/test/')
   val_paths    = data_ops.getPaths('datasets/'+DATA+'/val/')

   train_L, train_ab = loadImages(DATA)
   print len(trainB_paths),'training pairs'
   num_train = len(trainA_paths)
   num_test  = len(test_paths)
   num_val   = len(val_paths)

   n_critic = 1
   if LOSS_METHOD == 'wgan': n_critic = 5

   epoch_num = step/(num_train/BATCH_SIZE)

   while epoch_num < EPOCHS:
      s = time.time()
      epoch_num = step/(num_train/BATCH_SIZE)
      uiqms = []
      # pick random images every time for D
      for itr in xrange(n_critic):
         idx = np.random.choice(np.arange(num_train), BATCH_SIZE, replace=False)
         batch_L  = train_L[idx]
         batch_ab = train_ab[idx]
         sess.run(D_train_op, feed_dict={image_L:batch_L, image_ab:batch_ab})

      # also get new batch for G
      idx = np.random.choice(np.arange(num_train), BATCH_SIZE, replace=False)
      batch_L = train_L[idx]
      batch_ab = train_ab[idx]

      if AUGMENT:
         batch_L, batch_ab = data_ops.augment(batch_L, batch_ab)

      # calculate uiqm for each image generated by the generator - want to maximize this
      if UIQM_WEIGHT > 0.0:
         uiqm_gen_imgs = sess.run(gen_image, feed_dict={image_L:batch_L})
         for uimg in uiqm_gen_imgs:
            img_uiqm = uiqm.getUIQM(data_ops.deprocess(uimg))
            uiqms.append(img_uiqm)
         avg_uiqm = np.mean(uiqms)
         sess.run(G_train_op, feed_dict={image_L:batch_L, image_ab:batch_ab,uiqm_val:avg_uiqm})
         D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op], feed_dict={image_L:batch_L, image_ab:batch_ab,uiqm_val:avg_uiqm})
      else:
         sess.run(G_train_op, feed_dict={image_L:batch_L, image_ab:batch_ab})
         D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op], feed_dict={image_L:batch_L, image_ab:batch_ab})

      summary_writer.add_summary(summary, step)
      ss = time.time()-s

      if UIQM_WEIGHT > 0.0:
         print 'epoch:',epoch_num,'step:',step,'D loss:',D_loss,'G_loss:',G_loss,'UIQM:',avg_uiqm,'time:',ss
      else:
         print 'epoch:',epoch_num,'step:',step,'D loss:',D_loss,'G_loss:',G_loss,'time:',ss
      step += 1
      
      if step%100 == 0:
         print 'Saving model...'
         saver.save(sess, EXPERIMENT_DIR+'checkpoint-'+str(step))
         saver.export_meta_graph(EXPERIMENT_DIR+'checkpoint-'+str(step)+'.meta')
         print 'Model saved\n'
         idx = np.random.choice(np.arange(num_val), BATCH_SIZE, replace=False)
         batch_paths = val_paths[idx]
         batch_R = np.empty((BATCH_SIZE, 256,256,3))
         batch_L = np.empty((BATCH_SIZE, 256, 256, 1), dtype=np.float64)
         print 'Testing on val split...'
         j = 0
         for a in batch_paths:
            a_img = io.imread(a)
            a_img = transform.resize(a_img, (256,256,3))
            a_img = color.rgb2lab(a_img)
            img_L = a_img[:,:,0]
            img_L = np.expand_dims(img_L, 2)
            img_L = img_L/50.0-1.
            batch_L[j, ...]  = img_L
            batch_R[j,...] = transform.resize(io.imread(a), (256,256,3))
            j += 1

         gen_images = np.asarray(sess.run(gen_ab, feed_dict={image_L:batch_L}))
         c = 0
         val_uiqms = []
         for gen, real, L in zip(gen_images, batch_R, batch_L):
            gen = gen*128
            print gen
            gen = np.concatenate([L, gen], axis=2)
            gen = np.clip(gen, -128, 128)
            gen = color.lab2rgb(gen)
            img_uiqm = uiqm.getUIQM(data_ops.deprocess(gen))
            val_uiqms.append(img_uiqm)
            misc.imsave(IMAGES_DIR+str(step)+'_real.png', real)
            misc.imsave(IMAGES_DIR+str(step)+'_gen.png', gen)
            c += 1
            if c == 5: break
         print 'Done with val images, average uiqm:',np.mean(np.asarray(val_uiqms))
