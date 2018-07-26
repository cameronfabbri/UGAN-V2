import os
import sys
import glob
import ntpath
import random
import fnmatch
import numpy as np
from tqdm import tqdm
import cPickle as pickle
from shutil import copyfile

def getPaths(data_dir):
   pkl_paths = []
   pattern = '*.pkl'
   for d, s, fList in os.walk(data_dir):
      for filename in fList:
         if fnmatch.fnmatch(filename, pattern):
            fname_ = os.path.join(d,filename)
            pkl_paths.append(fname_)
   return np.asarray(pkl_paths)


if __name__ == '__main__':

   # search for all pkl files in checkpoints directory
   pkl_paths = getPaths('../checkpoints/')

   # get the list of all test images
   test_images_ = glob.glob('../datasets/underwater_imagenet/test/*.jpg')
   test_images  = []
   
   for t in test_images_:
      t = ntpath.basename(t).split('.jpg')[0]
      test_images.append(t)

   # get 4 random images
   image_names = random.sample(test_images, 4)

   # copy those 4 original distorted images here

   j = 0
   for p in pkl_paths:

      print p

      img_dir  = p.split('info.pkl')[0]+'test_images/gen/'
      pkl_file = open(p, 'rb')
      a = pickle.load(pkl_file)

      LOSS_METHOD   = a['LOSS_METHOD']
      NUM_LAYERS    = a['NUM_LAYERS']
      IG_WEIGHT     = a['IG_WEIGHT']

      if IG_WEIGHT == 1.0: ig_name = '-p'
      else: ig_name = ''

      # rename images
      new_images = []
      i = 0
      for image in image_names:
         new_name = str(i)+'_'+str(j)+'_'+LOSS_METHOD+'_'+str(NUM_LAYERS)+ig_name+'.png'
         new_images.append(new_name)
         i += 1

      # copy images from their dirs
      for og_im, n_im in zip(image_names,new_images):
         og_im = img_dir+og_im+'_gen.png'
         copyfile(og_im, n_im)

      j += 1
