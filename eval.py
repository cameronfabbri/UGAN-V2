'''

   Evaluation File

   Tests images from the test set of whatever dataset we used - in the pickle file

'''

import cPickle as pickle
import tensorflow as tf
from scipy import misc
from tqdm import tqdm
import numpy as np
import argparse
import random
import ntpath
import sys
import os
import time
import time
import glob
import cPickle as pickle
from tqdm import tqdm

sys.path.insert(0, 'ops/')
sys.path.insert(0, 'nets/')
from tf_ops import *

import data_ops

if __name__ == '__main__':

   if len(sys.argv) < 2:
      print 'You must provide an info.pkl file'
      exit()

   pkl_file = open(sys.argv[1], 'rb')
   a = pickle.load(pkl_file)

   LEARNING_RATE = a['LEARNING_RATE']
   LOSS_METHOD   = a['LOSS_METHOD']
   BATCH_SIZE    = a['BATCH_SIZE']
   L1_WEIGHT     = a['L1_WEIGHT']
   IG_WEIGHT     = a['IG_WEIGHT']
   NETWORK       = a['NETWORK']
   AUGMENT       = a['AUGMENT']
   EPOCHS        = a['EPOCHS']
   DATA          = a['DATA']

   EXPERIMENT_DIR  = 'checkpoints/LOSS_METHOD_'+LOSS_METHOD\
                     +'/NETWORK_'+NETWORK\
                     +'/L1_WEIGHT_'+str(L1_WEIGHT)\
                     +'/IG_WEIGHT_'+str(IG_WEIGHT)\
                     +'/AUGMENT_'+str(AUGMENT)\
                     +'/DATA_'+DATA+'/'\

   IMAGES_DIR     = EXPERIMENT_DIR+'test_images/'

   print
   print 'Creating',IMAGES_DIR
   try: os.makedirs(IMAGES_DIR)
   except: pass

   print
   print 'LEARNING_RATE: ',LEARNING_RATE
   print 'LOSS_METHOD:   ',LOSS_METHOD
   print 'BATCH_SIZE:    ',BATCH_SIZE
   print 'L1_WEIGHT:     ',L1_WEIGHT
   print 'IG_WEIGHT:     ',IG_WEIGHT
   print 'NETWORK:       ',NETWORK
   print 'EPOCHS:        ',EPOCHS
   print 'DATA:          ',DATA
   print

   if NETWORK == 'pix2pix': from pix2pix import *
   if NETWORK == 'resnet':  from resnet import *

   # global step that is saved with a model to keep track of how many steps/epochs
   global_step = tf.Variable(0, name='global_step', trainable=False)

   # underwater image
   image_u = tf.placeholder(tf.float32, shape=(1, 256, 256, 3), name='image_u')

   # generated corrected colors
   layers    = netG_encoder(image_u)
   gen_image = netG_decoder(layers)

   saver = tf.train.Saver(max_to_keep=1)

   init = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
   sess = tf.Session()
   sess.run(init)

   ckpt = tf.train.get_checkpoint_state(EXPERIMENT_DIR)
   if ckpt and ckpt.model_checkpoint_path:
      print "Restoring previous model..."
      try:
         saver.restore(sess, ckpt.model_checkpoint_path)
         print "Model restored"
      except:
         print "Could not restore model"
         pass
   
   # testing paths
   exts = ['*.jpg', '*.jpeg', '*.JPEG', '*.png']
   test_paths = []
   for ex in exts:
      test_paths.extend(glob.glob('datasets/'+DATA+'/test/'+ex))
   test_paths = np.asarray(test_paths)

   num_test = len(test_paths)

   print 'num test:',num_test
   print 'IMAGES_DIR:',IMAGES_DIR
   
   step = int(sess.run(global_step))
   times = []
   for img_path in tqdm(test_paths):

      img_name = ntpath.basename(img_path)

      img_name = img_name.split('.')[0]

      batch_images = np.empty((1, 256, 256, 3), dtype=np.float32)

      a_img = misc.imread(img_path).astype('float32')
      a_img = misc.imresize(a_img, (256, 256, 3))
      a_img = data_ops.preprocess(a_img)
      batch_images[0, ...] = a_img

      s = time.time()
      gen_images = np.asarray(sess.run(gen_image, feed_dict={image_u:batch_images}))
      tot = time.time()-s
      times.append(tot)

      for gen, real in zip(gen_images, batch_images):
         misc.imsave(IMAGES_DIR+img_name+'_real.png', real)
         misc.imsave(IMAGES_DIR+img_name+'_gen.png', gen)

   avg_time = float(np.mean(np.asarray(times)))
   print
   print 'average time:',avg_time
   print 'fps:',1.0/avg_time
   print
