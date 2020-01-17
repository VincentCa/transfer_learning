# Define global imports

import numpy as np
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, UpSampling2D, Concatenate, concatenate
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import os
import skimage
import skimage.io
import skimage.transform
import glob
import matplotlib.pyplot as plt
from IPython.display import clear_output
import cv2


class CustomDataGenerator(tf.keras.utils.Sequence):
    """Custom data generator that yields tuples of (image, mask) for a pre-processed version of the Pascal VOC 2012 dataset."""
    def __init__(self, source_raw, source_mask, filenames, batch_size, target_height, target_width, augmentation=True, full_resolution=False):
        self.source_raw = source_raw
        self.source_mask = source_mask
        self.filenames = filenames
        self.batch_size = batch_size
        self.target_height = target_height
        self.target_width = target_width
        self.augmentation = augmentation
        self.full_resolution = full_resolution
        self.on_epoch_end()

    def on_epoch_end(self):
        '''Shuffle list of files after each epoch.'''
        np.random.shuffle(self.filenames)
        
    def __getitem__(self, index):
        cur_files = self.filenames[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X, y = self.__data_generation(cur_files)
        return X, y

    def __data_generation(self, cur_files):
        X = np.empty(shape=(self.batch_size, self.target_height, self.target_width, 3))
        Y = np.empty(shape=(self.batch_size, self.target_height, self.target_width, 1), dtype=np.int32)
        for i, file in enumerate(cur_files):            
            img_raw = img_to_array(load_img(os.path.join(self.source_raw, file) + '.jpg', interpolation='bilinear', target_size=(256, 256))) 

            # The preprocessing function varies by architecture. 
            # e.g. for ResNet50, caffe-style preprocessing is used.
            # e.g. for MobileNetV2, tf-style preprocessing is used.
            img_raw = tf.keras.applications.mobilenet_v2.preprocess_input(img_raw) 
            
            # General note: people sometimes accidentally use bilinear interpolation when resizing masks.
            # If you need to resize, make sure to use nearest neighbor interpolation only to avoid invalid class labels.
            img_mask = np.load(os.path.join(self.source_mask, file) + '.npy')
      
            if self.augmentation:
                # Random cropping.
                crop_x = np.random.randint(img_raw.shape[1] - self.target_width)
                crop_y = np.random.randint(img_raw.shape[0] - self.target_height)
            else: # Take center crop instead.
                crop_x = (img_raw.shape[1] - self.target_width) // 2
                crop_y = (img_raw.shape[0] - self.target_height) // 2

            if not self.full_resolution:
              img_raw = img_raw[crop_y:crop_y+self.target_height, crop_x:crop_x+self.target_width]
              img_mask = img_mask[crop_y:crop_y+self.target_height, crop_x:crop_x+self.target_width]
                
            # Random flipping.
            perform_flip = np.random.rand(1) < 0.5
            if self.augmentation and perform_flip:
                img_raw = np.flip(img_raw, axis=1)
                img_mask = np.flip(img_mask, axis=1)

            X[i] = img_raw
            Y[i] = img_mask
        return X, Y
    
    def __len__(self):
        return int(np.floor(len(self.filenames) / self.batch_size))
   

def norm_vis(img, mode='rgb'):
    img_norm = (img - img.min()) / (img.max() - img.min())
    return img_norm if mode == 'rgb' else np.flip(img_norm, axis=2)








