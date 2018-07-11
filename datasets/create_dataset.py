'''

   Creates symlinks to the data we want to use

'''
import pickle
import fnmatch
import os
import ntpath
import random
from tqdm import tqdm

def getPaths(data_dir,ext='jpg'):
   pattern   = '*.'+ext
   image_paths = []
   for d, s, fList in os.walk(data_dir):
      for filename in fList:
         if fnmatch.fnmatch(filename, pattern):
            fname_ = os.path.join(d,filename)
            image_paths.append(fname_)
   return image_paths

if __name__ == '__main__':

   distorted_paths = getPaths('/mnt/data1/images/ugan_datasets/distorted/')

   '''
   i = 0
   for src in tqdm(distorted_paths):
      f_name = ntpath.basename(src)
      dest = 'large/test/'+str(i)+'_'+f_name
      try: os.symlink(src, dest)
      except: pass
      i += 1
   '''

   a = pickle.load(open('/mnt/data1/images/ugan_datasets/anything/labels.pkl','rb'))
   a = a[0]

   d_count = 0
   n_count = 0
   for key,val in a.items():
      if val == 1:
         d_count += 1
         distorted_paths.append(key)

   random.shuffle(distorted_paths)
   random.shuffle(distorted_paths)
   random.shuffle(distorted_paths)
   random.shuffle(distorted_paths)
   random.shuffle(distorted_paths)

   val_num = 20000
   val_paths = distorted_paths[:20000]
   test_paths = distorted_paths[20000:]

   i = 0
   for src in tqdm(val_paths):
      f_name = ntpath.basename(src)
      dest = 'large/val/'+str(i)+'_'+f_name
      try: os.symlink(src, dest)
      except: pass
      i += 1

   i = 0
   for src in tqdm(test_paths):
      f_name = ntpath.basename(src)
      dest = 'large/test/'+str(i)+'_'+f_name
      try: os.symlink(src, dest)
      except: pass
      i += 1
