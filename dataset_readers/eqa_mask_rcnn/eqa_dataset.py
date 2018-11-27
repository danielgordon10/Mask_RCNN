import pdb
import h5py
import glob
import os


import sys
import time
import numpy as np
import cv2
import scipy.misc
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".

import zipfile
import urllib.request
import shutil

# Root directory of the project
ROOT_DIR = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from dataset_readers.eqa_mask_rcnn import eqa_dataset

sys.path.append(os.path.join(ROOT_DIR, os.pardir))
import constants

USE_HDF5 = False

DATA_FOLDER = os.path.join(ROOT_DIR, 'data/eqa/train/')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs/eqa")

############################################################
#  Configurations
############################################################

IMAGE_WIDTH = 448
IMAGE_HEIGHT = 448

class EQAConfig(Config):
    global MODEL_DIR
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "eqa"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 8 #constants.PARALLEL_SIZE

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8
    #BACKBONE = "mobilenet224v1"
    BACKBONE = "resnet101"

    # Number of classes (including background)
    NUM_CLASSES = len(constants.OBJECTS) # COCO has 80 classes
    IMAGE_MIN_DIM = IMAGE_WIDTH
    IMAGE_MAX_DIM = IMAGE_WIDTH

    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 50
    START_LINEAR_INCREASE = 90
    END_LINEAR_INCREASE = 200
    END_LINEAR_DECREASE = 300
    MIN_LR = 0.001
    MAX_LR = 0.0025
    TRAIN_ROIS_PER_IMAGE = 50
    DETECTION_MAX_INSTANCES = 20
    POST_NMS_ROIS_INFERENCE = 100
    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_NMS_THRESHOLD = 0.3

    MODEL_DIR = os.path.join(MODEL_DIR, BACKBONE)


############################################################
#  Dataset
############################################################


class EQADataset(utils.Dataset):
    def __init__(self):
        super(EQADataset, self).__init__()
        if USE_HDF5:
            data_file_name = glob.glob(DATA_FOLDER + '*.h5')[-1]
            self.dataset = h5py.File(data_file_name, 'r')
            self.num_entries = len(self.dataset['images/seg_images'])
        else:
            self.image_names = sorted(glob.glob(os.path.join(DATA_FOLDER, 'color_images', '*.jpg')))
            #self.label_names = sorted(glob.glob(os.path.join(DATA_FOLDER, 'seg_images', '*.png')))
            #self.class_labels = sorted(glob.glob(os.path.join(DATA_FOLDER, 'class_labels', '*.npy')))
            #self.class_labels = np.load(os.path.join(DATA_FOLDER, 'class_labels.npy'))
            #self.num_objects = np.load(os.path.join(DATA_FOLDER, 'num_objects.npy'))
            self.num_entries = len(self.image_names)
            #self.angle = np.load(os.path.join(DATA_FOLDER, 'angle.npy'))

        self.num_entries_train = int(self.num_entries)

    def __len__(self):
        return self.num_entries

    def load_data(self, subset):
        for ii, obj in enumerate(constants.OBJECTS):
            if obj != 'background':
                self.add_class('eqa', ii, obj)
        if subset == 'train':
            for ii in range(self.num_entries_train):
                self.add_image(
                        'eqa', image_id=ii,
                        path='',
                        width=IMAGE_WIDTH,
                        height=IMAGE_HEIGHT)
        elif subset in {'val_seen', 'val_unseen'}:
            DATA_FOLDER = os.path.join(ROOT_DIR, 'data/eqa/%s/' % subset)
            self.image_names = sorted(glob.glob(os.path.join(DATA_FOLDER, 'color_images', '*.jpg')))
            #self.label_names = sorted(glob.glob(os.path.join(DATA_FOLDER, 'seg_images', '*.png')))
            #self.class_labels = sorted(glob.glob(os.path.join(DATA_FOLDER, 'class_labels', '*.npy')))
            #self.class_labels = np.load(os.path.join(DATA_FOLDER, 'class_labels.npy'))
            #self.num_objects = np.load(os.path.join(DATA_FOLDER, 'num_objects.npy'))
            self.num_entries = len(self.image_names)
            for ii in range(self.num_entries):
                self.add_image(
                        'eqa', image_id=ii,
                        path='',
                        width=IMAGE_WIDTH,
                        height=IMAGE_HEIGHT)
        else:
            raise Error('Unknown data split %s' % subset)

    def load_image(self, image_id):
        image_id = self.image_info[image_id]['id']
        if USE_HDF5:
            return self.dataset['images/color_images'][image_id, ...]
        else:
            #return (cv2.imread(self.image_names[image_id])[:, :, ::-1], self.angle[image_id])
            return (cv2.imread(self.image_names[image_id])[:, :, ::-1], 0)

    def load_mask(self, image_id):
        image_id = self.image_info[image_id]['id']
        if USE_HDF5:
            mask = self.dataset['images/seg_images'][image_id, ...]
            num_objects = self.dataset['images/num_objects'][image_id, ...].squeeze()
            classes = self.dataset['images/class_labels'][image_id, ...]
            classes = classes[1:num_objects + 1].astype(np.int32)
        else:
            label_name = self.image_names[image_id].replace('color_images', 'seg_images')[:-4] + '.png'
            mask = cv2.imread(label_name)[:, :, 0]  # Opencv reads 3 channel, but they're all the same.
            class_label_name = self.image_names[image_id].replace('color_images', 'class_labels')[:-4] + '.npy'
            class_labels = np.load(class_label_name)
            num_objects = class_labels[0]
            classes = class_labels[2:2 + num_objects].astype(np.int32)
            #num_objects = self.num_objects[image_id].squeeze()
            #classes = self.class_labels[image_id, ...]
        mask = np.eye(num_objects + 1)[mask][:, :, 1:]
        return mask, classes

    def image_reference(self, image_id):
        return image_id






############################################################
#  Training
############################################################

def main(args):
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on EQA.')
    parser.add_argument('--dataset',
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--model',
                        default='last',
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=MODEL_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--gpus', required=False,
                        default='1',
                        help='Number of gpus to use')

    args = parser.parse_args(args)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    gpu_count = len(args.gpus.split(','))

    # Configurations
    config = EQAConfig()
    config.GPU_COUNT = gpu_count
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)

    # Train or evaluate
    try:
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = eqa_dataset.EQADataset()
        dataset_train.load_data('train')
        dataset_train.prepare()

        # Validation dataset
        dataset_val = eqa_dataset.EQADataset()
        dataset_val.load_data('val_unseen')
        dataset_val.prepare()

        # Image Augmentation
        # Right/Left flip 50% of the time
        augmentation = imgaug.augmenters.Fliplr(0.5)

        # *** This training schedule is an example. Update to your needs ***

        '''
        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads',
                    augmentation=augmentation)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=120,
                    layers='4+',
                    augmentation=augmentation)
        '''

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=350,
                    layers='all',
                    augmentation=augmentation)
    except:
        import traceback
        traceback.print_exc()
        print('\nSaving model')
        model.save_model()
        print('Saved model')

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
