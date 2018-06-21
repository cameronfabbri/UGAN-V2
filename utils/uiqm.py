'''

   Computes the Underwater Image Quality Measure (UIQM)

'''
from skimage.util.shape import view_as_windows
from skimage.util.shape import view_as_blocks
from scipy import ndimage
from scipy import misc
from PIL import Image
import numpy as np
import math
import sys
import cv2

'''
   Calculates the asymetric alpha-trimmed mean
'''
def mu_a(x, alpha_L=0.1, alpha_R=0.1):

   # sort pixels by intensity - for clipping
   x = sorted(x)

   # get number of pixels
   K = len(x)

   # calculate T alpha L and T alpha R
   T_a_L = math.ceil(alpha_L*K)
   T_a_R = math.floor(alpha_R*K)

   # calculate mu_alpha weight
   weight = (1/(K-T_a_L-T_a_R))

   # loop through flattened image starting at T_a_L+1 and ending at K-T_a_R
   s   = int(T_a_L+1)
   e   = int(K-T_a_R)
   val = sum(x[s:e])
   val = weight*val
   return val

def s_a(x, mu):

   val = 0
   for pixel in x:
      val += math.pow((pixel-mu), 2)

   return val/len(x)

def _uicm(x):

   R = x[:,:,0].flatten()
   G = x[:,:,1].flatten()
   B = x[:,:,2].flatten()

   RG = R-G
   YB = ((R+G)/2)-B

   mu_a_RG = mu_a(RG)
   mu_a_YB = mu_a(YB)

   print 'mu_a_RG:',mu_a_RG
   print 'mu_a_YB:',mu_a_YB

   s_a_RG = s_a(RG, mu_a_RG)
   s_a_YB = s_a(YB, mu_a_YB)

   print 's_a_RG:',s_a_RG
   print 's_a_YB:',s_a_YB

   l = math.sqrt( (math.pow(mu_a_RG,2)+math.pow(mu_a_YB,2)) )
   r = math.sqrt(s_a_RG+s_a_YB)

   return (-0.0268*l)+(0.1586*r)

def sobel(x):
   dx = ndimage.sobel(x,0)
   dy = ndimage.sobel(x,1)
   mag = np.hypot(dx, dy)
   mag *= 255.0 / np.max(mag) 
   return mag

'''
   Enhancement measure estimation

   x.shape[0] = height
   x.shape[1] = width

'''
def eme(x, window_size):

   # if 4 blocks, then 2x2...etc.
   k1 = x.shape[1]/window_size
   k2 = x.shape[0]/window_size

   w = 2./(k1*k2)

   blocksize_x = window_size
   blocksize_y = window_size

   #print 'blocksize_x:',blocksize_x
   #print 'blocksize_y:',blocksize_y

   # make sure image is divisible by window_size
   #print blocksize_x*k1
   #print blocksize_y*k2
   x = x[:blocksize_y*k2, :blocksize_x*k1]

   start_x = 0
   start_y = 0
   val = 0
   for l in range(k1):
      for k in range(k2):

         #end_x = start_x + blocksize_x
         #end_y = start_y + blocksize_y

         # find max and min of block
         block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1)]
         #print 'x,y:',x.shape[1], x.shape[0]
         #print 'block_shape:',block.shape
         max_ = np.max(block)
         min_ = np.min(block)

         #print max_
         #print min_

         if min_ == 0.0: val += 0
         elif max_ == 0.0: val += 0
         else: val += math.log(max_/min_)

   print w*val

   exit()

   weight = 2/(k1*k2)

'''
   Underwater Image Sharpness Measure
'''
def _uism(x):
   
   R = x[:,:,0]
   G = x[:,:,1]
   B = x[:,:,2]

   # first apply Sobel edge detector to each RGB component
   Rs = sobel(R)
   Gs = sobel(G)
   Bs = sobel(B)

   R_edge_map = np.multiply(Rs, R)
   G_edge_map = np.multiply(Gs, G)
   B_edge_map = np.multiply(Bs, B)

   #misc.imsave('R_edge_map.png', R_edge_map)
   #misc.imsave('G_edge_map.png', G_edge_map)
   #misc.imsave('B_edge_map.png', B_edge_map)

   r_eme = eme(R_edge_map, 10)
   g_eme = eme(G_edge_map, 10)
   b_eme = eme(B_edge_map, 10)

   exit()

if __name__ == '__main__':

   if len(sys.argv) < 2:
      print 'Usage: python uiqm.py [image]'
      exit()

   image = cv2.imread(sys.argv[1])
   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   #image = cv2.resize(image, (256,256))
   image = image.astype(np.float32)

   #uicm = _uicm(image)
   uism = _uism(image)

   print 'UICM:',uicm
