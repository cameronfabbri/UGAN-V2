import tensorflow as tf
import sys
import scipy.misc as misc
import numpy as np
sys.path.insert(0, 'ops/')
sys.path.insert(0, 'measures/')
from uiqm import *
from data_ops import *
import cv2

img = cv2.imread('temp.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.expand_dims(img, 0)
img_p = preprocess(img)
