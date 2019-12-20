#!/usr/bin/env python
# coding: utf-8

# # Image segmentaion with masked facies
# 
# Squish rectangular images to square

# In[1]:



# In[2]:


import sys
from numbers import Integral
from random import uniform
from PIL import Image as pil_image
import fastai
from fastai.vision import *
from fastai.vision import Image
from fastai.vision.transform import _minus_epsilon
from fastai.vision.data import SegmentationProcessor
from fastai.vision.interpret import SegmentationInterpretation
from mask_functions import *
from collections import defaultdict
import cv2
from IPython.display import display 
import datetime
import uuid


# In[3]:


fastai.__version__


# In[4]:


# In[8]:


tgt_height = 256
data_dir = Path('data')
train_images = data_dir/'train_images'
test_img = train_images/'mask_fill/test'

train_path = train_images/'mask_fill/train'
train_mask = train_path/'masks'
train_img = train_path/'images'


# ### Data

# In[13]:


train_img_names = get_image_files(train_img)
len(train_img_names)



train_mask_names = get_image_files(train_mask)
train_mask_names[:3]




# ### Convert masks to n color values only



u_values=[]
def get_all_uniques(images):
    for i, im in enumerate(images):
        if (i % 100)==0:
            print(f'converting {i}, {im}')
        mask = np.asarray(pil_image.open(im))
        u_values.extend(np.unique(mask.tolist()))
    return u_values


# In[ ]:


u_values=list(set(get_all_uniques(train_mask_names)))
u_values.sort()


# In[ ]:


print(f'u_values: {u_names}')

