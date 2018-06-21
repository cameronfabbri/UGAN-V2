'''

   Computes the Underwater Image Quality Measure (UIQM)

'''

from scipy import misc
import numpy as np
import math
import sys
import cv2
from PIL import Image
from scipy import ndimage

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
'''
def eme():

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

   edge_map = sobel(x)

   out = edge_map*x

   misc.imsave('out.png', out)

   exit()

if __name__ == '__main__':

   if len(sys.argv) < 2:
      print 'Usage: python uiqm.py [image]'
      exit()

   image = cv2.imread(sys.argv[1])
   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   image = image.astype(np.float32)

   uicm = _uicm(image)
   uism = _uism(image)

   print 'UICM:',uicm
