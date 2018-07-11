'''

Operations used for data management

'''

from __future__ import division
from __future__ import absolute_import

from scipy import misc
from skimage import color
import collections
import tensorflow as tf
import numpy as np
import math
import time
import random
import glob
import os
import fnmatch
import cPickle as pickle
import cv2

# [-1,1] -> [0, 255]
def deprocess(x):
   return (x+1.0)*127.5

# [0,255] -> [-1, 1]
def preprocess(x):
   return (x/127.5)-1.0

def preprocess_lab(lab):
    with tf.name_scope('preprocess_lab'):
        L_chan, a_chan, b_chan = tf.unstack(lab, axis=2)
        # L_chan: black and white with input range [0, 100]
        # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
        # [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
        return [L_chan / 50 - 1, a_chan / 110, b_chan / 110]


def deprocess_lab(L_chan, a_chan, b_chan):
    with tf.name_scope('deprocess_lab'):
        # this is axis=3 instead of axis=2 because we process individual images but deprocess batches
        return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=3)
        #return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=2)

'''
def augment(image, brightness):
    # (a, b) color channels, combine with L channel and convert to rgb
    a_chan, b_chan = tf.unstack(image, axis=3)
    L_chan = tf.squeeze(brightness, axis=3)
    lab = deprocess_lab(L_chan, a_chan, b_chan)
    rgb = lab_to_rgb(lab)
    return rgb
'''

'''
   Augment images - a is distorted
'''
def augment(a_img, b_img):

   # randomly interpolate
   a = random.random()
   a_img = a_img*(1-a) + b_img*a

   # kernel for gaussian blurring
   kernel = np.ones((5,5),np.float32)/25
   
   # flip image left right
   r = random.random()
   if r < 0.5:
      a_img = np.fliplr(a_img)
      b_img = np.fliplr(b_img)
   
   # flip image up down
   r = random.random()
   if r < 0.5:
      a_img = np.flipud(a_img)
      b_img = np.flipud(b_img)
   
   # send in the clean image for both
   r = random.random()
   if r < 0.5:
      a_img = b_img

   # perform some gaussian blur on distorted image
   r = random.random()
   if r < 0.5:
      a_img = cv2.filter2D(a_img,-1,kernel)

   # resize to 286x286 and perform a random crop
   r = random.random()
   if r < 0.5:
      a_img = misc.imresize(a_img, (286, 286,3))
      b_img = misc.imresize(b_img, (286, 286,3))

      rand_x = random.randint(0,50)
      rand_y = random.randint(0,50)

      a_img = a_img[rand_x:, rand_y:, :]
      b_img = b_img[rand_x:, rand_y:, :]

      a_img = misc.imresize(a_img, (256,256,3))
      b_img = misc.imresize(b_img, (256,256,3))

   return a_img, b_img

def getPaths(data_dir):
   exts = ['*.png','*.PNG','*.jpg','*.JPG', '*.JPEG']
   image_paths = []
   for pattern in exts:
      for d, s, fList in os.walk(data_dir):
         for filename in fList:
            if fnmatch.fnmatch(filename, pattern):
               fname_ = os.path.join(d,filename)
               image_paths.append(fname_)
   return np.asarray(image_paths)


# TODO add in files to exclude (gray ones)
def loadData(batch_size, train=True):

   print 'Reading data...'
   utrain_dir  = '/mnt/data2/images/underwater/youtube/'
   #places2_dir  = '/mnt/data2/images/underwater/youtube/'
   #places2_dir    = '/mnt/data2/images/places2/test2014/'
   places2_dir    = '/mnt/data2/images/places2_standard/train_256/'

   # get all paths for underwater images
   u_paths    = getPaths(utrain_dir)
   # get all paths for places2 images
   places2_paths = getPaths(places2_dir)

   # shuffle all the underwater paths before splitting into test/train
   random.shuffle(u_paths)

   # shuffle places2 images because why not
   random.shuffle(places2_paths)

   # take 90% for train, 10% for test
   train_num = int(0.95*len(u_paths))
   utrain_paths = u_paths[:train_num]
   utest_paths  = u_paths[train_num:]

   # write train underwater data
   pf   = open(pkl_utrain_file, 'wb')
   data = pickle.dumps(utrain_paths)
   pf.write(data)
   pf.close()

   # write test underwater data
   pf   = open(pkl_utest_file, 'wb')
   data = pickle.dumps(utest_paths)
   pf.write(data)
   pf.close()

   # write test places2 data
   pf   = open(pkl_places2_file, 'wb')
   data = pickle.dumps(places2_paths)
   pf.write(data)
   pf.close()

   if train:
      print
      print len(utrain_paths), 'underwater train images'
      print len(utest_paths), 'underwater test images'
      print len(places2_paths), 'places2 images'
      print

   if train: upaths = utrain_paths
   else: upaths = utest_paths

   decode = tf.image.decode_image

   # load underwater images
   with tf.name_scope('load_underwater'):
      path_queue = tf.train.string_input_producer(upaths, shuffle=train)
      reader = tf.WholeFileReader()
      paths, contents = reader.read(path_queue)
      raw_input_ = decode(contents)
      raw_input_ = tf.image.convert_image_dtype(raw_input_, dtype=tf.float32)

      raw_input_.set_shape([None, None, 3])

      # randomly flip image if training
      seed = random.randint(0, 2**31 - 1) 
      if train: raw_input_ = tf.image.random_flip_left_right(raw_input_, seed=seed)
      
      # convert to LAB and process gray channel and color channels
      lab = rgb_to_lab(raw_input_)
      L_chan, a_chan, b_chan = preprocess_lab(lab)
      gray_images = tf.expand_dims(L_chan, axis=2)   # shape (?,?,1)
      ab_images = tf.stack([a_chan, b_chan], axis=2) # shape (?,?,2)
      
      gray_images = tf.image.resize_images(gray_images, [256, 256], method=tf.image.ResizeMethod.AREA)
      ab_images   = tf.image.resize_images(ab_images, [256, 256], method=tf.image.ResizeMethod.AREA)

      u_paths_batch, gray_batch, ab_batch = tf.train.shuffle_batch([paths, gray_images, ab_images], batch_size=batch_size, num_threads=8, min_after_dequeue=int(0.1*100), capacity=int(0.1*100)+8*batch_size)

   if train:
      # load the places2 images
      with tf.name_scope('load_places2'):
         path_queue = tf.train.string_input_producer(places2_paths, shuffle=True)
         reader = tf.WholeFileReader()
         paths, contents = reader.read(path_queue)
         raw_input_ = decode(contents)
         raw_input_ = tf.image.convert_image_dtype(raw_input_, dtype=tf.float32)

         raw_input_.set_shape([None, None, 3])

         # randomly flip image
         seed = random.randint(0, 2**31 - 1) 
         raw_input_ = tf.image.random_flip_left_right(raw_input_, seed=seed)
         
         # convert to LAB
         places2_images = rgb_to_lab(raw_input_)
         L_chan, a_chan, b_chan = preprocess_lab(places2_images)
         gray_images = tf.expand_dims(L_chan, axis=2)   # shape (?,?,1)
         ab_images = tf.stack([a_chan, b_chan], axis=2) # shape (?,?,2)

         places2_L  = tf.image.resize_images(gray_images, [256,256], method=tf.image.ResizeMethod.AREA)
         places2_ab = tf.image.resize_images(ab_images, [256,256], method=tf.image.ResizeMethod.AREA)

         places2_paths_batch, places2_L_batch, places2_ab_batch = tf.train.shuffle_batch([paths, places2_L, places2_ab], batch_size=batch_size, num_threads=8, min_after_dequeue=int(0.1*100), capacity=int(0.1*100)+8*batch_size)
         
   else: places2_L_batch = places2_ab_batch = None

   return Data(
      ugray=gray_batch,
      uab=ab_batch,
      places2_L=places2_L_batch,
      places2_ab=places2_ab_batch
   )
