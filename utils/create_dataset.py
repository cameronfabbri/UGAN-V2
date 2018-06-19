'''
   Must be run from python 3

'''

import pickle
import numpy as np

if __name__ == '__main__':

   data = pickle.load(open('/mnt/data1/images/ugan_datasets/anything/labels.pkl', 'rb'))

   data_dict = data[0]

   count = 0
   for key,val in data_dict.items():
      # distorted
      if val == 1:
         #print('1:',key)
         count += 1
      # not distorted
      if val == 2:
         #print('2:',key)
         count += 1

   print('count:',count)
