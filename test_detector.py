import pdb
import os
import numpy as np
import tensorflow as tf
from collections import defaultdict
from utils import tf_util
from Mask_RCNN.detector import ObjectDetector
import mrcnn.model as modellib
import sys
ROOT_DIR = os.path.dirname(__file__)
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from dataset_readers.ai2thor_mask_rcnn import ai2thor_dataset
from dataset_readers.eqa_mask_rcnn import eqa_dataset
import tqdm
import constants

DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
if constants.EQA:
    DATASET = 'eqa'
    config = eqa_dataset.EQASingleInferenceConfig()
else:
    DATASET = 'ai2thor'
    config = ai2thor_dataset.AI2THORConfig()
if constants.EQA:
    model_path = 'eqa'
else:
    model_path = 'ai2thor'
MODEL_DIR = os.path.join(ROOT_DIR, 'logs', model_path, config.BACKBONE)
print('MODEL_DIR', MODEL_DIR)

with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config, create_log_dir=False)
# Set weights file path
weights_path = model.find_last()

# Or, uncomment to load the last model you trained
# weights_path = model.find_last()

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)
print("Weights loaded")

#for SPLIT in ['train', 'val_seen', 'val_unseen']:
for SPLIT in ['val_seen']:
    if constants.EQA:
        dataset = eqa_dataset.EQADataset()
    else:
        dataset = ai2thor_dataset.AI2THORDataset()
    dataset.load_data(SPLIT)
    # Must call before using the dataset
    dataset.prepare()

    print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

    results = defaultdict(list)
    APs = []
    precisions = []
    recalls = []
    overlaps = []
    for image_id in tqdm.tqdm(dataset.image_ids[:100]):
        pdb.set_trace()
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        angle = modellib.parse_image_meta(image_meta[np.newaxis, :])['image_angle']
        result = model.detect([image], angle)[0]
        results['rois'].append(result['rois'])
        results['class_ids'].append(result['class_ids'])
        results['scores'].append(result['scores'])
        results['masks'].append(result['masks'])
        _, AP = utils.compute_ap_range(gt_bbox, gt_class_id, gt_mask,
                result['rois'], result['class_ids'], result['scores'], result['masks'],
                np.arange(0.3, 1.0, 0.05), verbose=False)
        APs.append(AP)
        '''
        precisions.append(precision)
        recalls.append(recall)
        overlaps.append(overlap)
        '''

    APs = np.stack(APs, axis=0)
    APs = np.mean(APs, axis=0)
    result_file = open('Mask_RCNN/results_%s_%s.csv' % (DATASET, SPLIT), 'w')
    result_file.write(', '.join(['%.3f' % val for val in np.arange(0.3, 1.0, 0.05).tolist()]) + '\n')
    result_file.write(', '.join(['%.3f' % val for val in APs.tolist()]) + '\n')


