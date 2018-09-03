import os 
import sys
import random
import numpy as np
import cv2
from imgaug import augmenters as iaa
import tensorflow as tf
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

ROOT_DIR = os.getcwd()
DATA_DIR = ROOT_DIR+'/train'
ROOT_MODEL = ROOT_DIR+'/models/classic-500'

sys.path.append(os.path.join(ROOT_DIR, 'Mask_RCNN'))

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

## ---------------------- function ----------------------

def masks_as_image(in_mask_list, all_masks=None):
    if all_masks is None:
        all_masks = np.zeros((768, 768), dtype = np.int16)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)

def rle_decode(mask_rle, shape=(768, 768)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


## ---------------------- Config ----------------------

class DetectorConfig(Config):
    NAME = 'Airbus'
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 3
    
    NUM_CLASSES = 2  # background + 1 pneumonia classes
   
    IMAGE_MIN_DIM = 500
    IMAGE_MAX_DIM = 500
    
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 10


    
class DetectorDataset(utils.Dataset):
    def __init__(self, image_fps, image_annotations, orig_height, orig_width):
        super().__init__(self)
        
        # Add classes
        self.add_class('airbus', 1, 'ship')
   
        # add images 
        for i, fp in enumerate(image_fps):
            annotations = image_annotations.item()[fp]
            self.add_image('airbus', image_id=i, path=fp, 
                           annotations=annotations, orig_height=orig_height, orig_width=orig_width)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

    def load_image(self, image_id):
        info = self.image_info[image_id]
        fp = info['path']
        image = cv2.imread("train/"+fp)
        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotations = info['annotations']
        count = len(annotations)
        mask = np.zeros((info['orig_height'], info['orig_width'], count), dtype=np.uint8)
        for i, a in enumerate(annotations):
            imgmax = masks_as_image([a])
            mask[:, :, i] = imgmax[:,:,0]

        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)


    
    

# ----------  main  -------------
ORIG_SIZE = 768
config = DetectorConfig()

image_fps = np.load("images_ids.npy")
image_annotations = np.load("images_annotations.npy")

image_fps_train, image_fps_val = train_test_split(image_fps, test_size=0.1)


# prepare the training dataset
dataset_train = DetectorDataset(image_fps_train, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_train.prepare()


# prepare the validation dataset
dataset_val = DetectorDataset(image_fps_val, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_val.prepare()


augmentation = iaa.Sequential([
    iaa.Sometimes(0.50, iaa.Fliplr(0.5)),
    iaa.Sometimes(0.50, iaa.Flipud(0.5)),
    iaa.Sometimes(0.30, iaa.CoarseSalt(p=0.10, size_percent=0.02)),
    #iaa.Sometimes(0.30, iaa.Affine(rotate=(-25, 25))),
    iaa.Sometimes(0.30, iaa.GaussianBlur((0, 3.0)))
])

#### MODEL
model = modellib.MaskRCNN(mode='training', 
                          config=config, 
                          model_dir=ROOT_MODEL)


NUM_EPOCHS = 400
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=NUM_EPOCHS, 
            layers='all',
            augmentation=augmentation)

