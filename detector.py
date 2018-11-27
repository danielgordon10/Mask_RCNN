import pdb
import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import scipy.misc
import tensorflow as tf
from utils import drawing
from utils import py_util
from utils import tf_util


# Root directory of the project
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
sys.path.append(os.path.join(ROOT_DIR, os.path.pardir))
from dataset_readers.ai2thor_mask_rcnn import ai2thor_dataset
from dataset_readers.eqa_mask_rcnn import eqa_dataset
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn.model import log
import threading
import multiprocessing

import constants

if constants.USE_MULTIPROCESSING:
    from multiprocessing import Queue
else:
    try:
        from queue import Queue
    except ImportError:
        from Queue import Queue

# Directory to save logs and trained model

if constants.EQA:
    model_path = 'eqa'
else:
    model_path = 'ai2thor'
MODEL_DIR = os.path.join(ROOT_DIR, 'logs', model_path)


class ObjectDetector(object):
    def __init__(self, sess, devices='/gpu:0'):
        if constants.EQA:
            self.config = eqa_dataset.EQAConfig()
        else:
            self.config = ai2thor_dataset.AI2THORConfig()
        self.config.display()
        self.sess = sess
        self.devices = devices
        self.num_devices = len(self.devices.split(','))
        self.models = None
        self.batch_size = constants.MASK_RCNN_BATCH_SIZE
        self.graph = tf.get_default_graph()
        self.input_queue = Queue()
        self.output_queues = {}  # each thread should add their own output queue keyed to their thread_id
        self.exit_flag = False
        self.weights_loaded = False
        self.start_detector()

    def create_net(self):
        if self.devices is None:
            self.devices = '/gpu:0'
        devices = self.devices.split(',')
        self.models = []
        for device in devices:
            with tf.device(device):
                model = modellib.MaskRCNN(mode='inference', model_dir=os.path.join(MODEL_DIR, self.config.BACKBONE),
                                          config=self.config, sess=self.sess, create_log_dir=False)
                model.keras_model._make_predict_function()
            self.models.append(model)

    def load_weights(self):
        for model in self.models:
            weights_path = model.find_last()
            # Load weights
            print('Mask RCNN Loading weights ', weights_path)
            model.load_weights(weights_path, by_name=True)
        self.weights_loaded = True


    def add_user(self):
        output_queue = Queue()
        thread_id = len(self.output_queues)
        self.output_queues[thread_id] = output_queue
        return thread_id, output_queue

    def enqueue_data(self, image, thread_id):
        self.input_queue.put((thread_id, image))

    def start_detector(self):
        def run_thread(model_ind):
            while not self.weights_loaded:
                time.sleep(0.1)

            num_its = 0
            detection_count = 0
            total_detection_time = 0
            moving_mean_detection_time = None
            with self.graph.as_default():
                while not self.exit_flag:
                    num_its += 1
                    ids = []
                    #images = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(self.batch_size)]
                    images = []

                    current_id, current_image = self.input_queue.get()
                    ids.append(current_id)
                    images.append(current_image)

                    size = 1
                    while size < self.batch_size and not self.input_queue.empty():
                        current_id, current_image = self.input_queue.get()
                        ids.append(current_id)
                        images.append(current_image)
                        size += 1

                    t_start = time.time()
                    results = self.detect(model_ind, images)
                    for ii in range(size):
                        self.output_queues[ids[ii]].put(results[ii])

                    detection_time = time.time() - t_start

                    if moving_mean_detection_time is None:
                        moving_mean_detection_time = -1  # let it burn in a single run for allocation and whatnot.
                    elif moving_mean_detection_time == -1:
                        moving_mean_detection_time = detection_time / size
                    else:
                        moving_mean_detection_time = moving_mean_detection_time * 0.9 + detection_time / size * 0.1
                        total_detection_time += detection_time
                        detection_count += size

                    if num_its % 10 == 0:
                        #print('running Mask RCNN of size', size, ids, py_util.get_time_str())
                        print('--- Thread %d, Total detection time --- %.3f, # images: %d, current: %.3f, moving mean: %.3f, global mean: %.3f' %
                              (model_ind, detection_time, size, detection_time / size, moving_mean_detection_time,
                               (total_detection_time / max(1, detection_count))))

        for tt in range(self.num_devices):
            thread = threading.Thread(target=run_thread, args=(tt,))
            thread.daemon = True
            thread.start()

    def stop_detector(self):
        self.exit_flag = True

    def detect(self, model_ind, images, confidence_threshold=constants.DETECTION_THRESHOLD):
        if type(images) != list:
            images = list(images)
        results = self.models[model_ind].detect(images, verbose=False)

        unpacked_results = []
        for result in results:
            filter_inds = result['scores'] > confidence_threshold
            boxes = result['rois'][filter_inds]
            class_names = np.array([constants.OBJECTS[ii] for ii in result['class_ids'][filter_inds]])
            masks = result['masks'].transpose(2, 0, 1)[filter_inds]
            scores = result['scores'][filter_inds]

            # Display results
            boxes = boxes[:, [1, 0, 3, 2]]  # yxyx -> xyxy
            unpacked_results.append((boxes, scores, class_names, masks))
        return unpacked_results

    @staticmethod
    def visualize_detections(image, boxes, scores, classes, masks=None):
        '''
        if masks is not None:
            masks = masks.transpose(1, 2, 0)
            boxes = boxes[:, [1, 0, 3, 2]].copy()  # xyxy -> yxyx
            classes = np.array([constants.OBJECT_CLASS_TO_ID[cls] for cls in classes])
            detection_image = visualize.display_instances(image, boxes, masks, classes, constants.OBJECTS, scores,
                                                          plot=False)
            return detection_image
        else:
        '''
        out_image = image.copy()
        out_image = cv2.resize(out_image, (constants.SCREEN_WIDTH, constants.SCREEN_HEIGHT))

        #if masks is None and len(boxes) > 0:
            #boxes = (boxes / np.array([constants.SCREEN_HEIGHT * 1.0 / image.shape[1],
                                       #constants.SCREEN_WIDTH * 1.0 / image.shape[0]])[[0, 1, 0, 1]]).astype(np.int32)
        for ii, box in enumerate(boxes):
            mask = None if masks is None else masks[ii]
            drawing.draw_detection_box(out_image, box, classes[ii], confidence=scores[ii], width=2, mask=mask)
        return out_image

def main():
    # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
    os.environ['CUDA_VISIBLE_DEVICES'] = constants.GPU_ID
    import glob
    DATASET = 'eqa' if constants.EQA else 'ai2thor'
    TYPE = 'val_unseen'
    print('ROOT DIR', ROOT_DIR)
    PATH_TO_TEST_IMAGES_DIR = '%s/data/%s/%s/color_images/' % (ROOT_DIR, DATASET, TYPE)
    print('path', PATH_TO_TEST_IMAGES_DIR)
    TEST_IMAGE_PATHS = sorted(glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, '*.jpg')))
    NUM_IMAGES = 100
    TEST_IMAGE_PATHS = TEST_IMAGE_PATHS[:min(len(TEST_IMAGE_PATHS), NUM_IMAGES)][::-1]
    TEST_IMAGE_PATHS = list(enumerate(TEST_IMAGE_PATHS))
    print('num images', len(TEST_IMAGE_PATHS))
    TEST_IMAGE_PATHS_QUEUE = Queue()
    for im in TEST_IMAGE_PATHS:
        TEST_IMAGE_PATHS_QUEUE.put(im)

    test_base = '%s/test_images/%s/%s/' % (ROOT_DIR, DATASET, TYPE)
    if not os.path.exists(test_base + 'output'):
        os.makedirs(test_base + 'output')
    if not os.path.exists(test_base + 'input'):
        os.makedirs(test_base + 'input')

    from utils import tf_util
    sess = tf_util.Session()

    object_detector = ObjectDetector(sess=sess, devices=constants.DETECTION_DEVICE)

    def run_detector(object_pipes):
        thread_id, input_queue, output_queue = object_pipes
        while not TEST_IMAGE_PATHS_QUEUE.empty():
            try:
                image_id, image_path = TEST_IMAGE_PATHS_QUEUE.get(timeout=0.001)
                image = cv2.imread(image_path)[:, :, ::-1]
                input_queue.put((thread_id, image))
                boxes, scores, classes, masks = output_queue.get()
                if constants.DEBUG:
                    output_image = ObjectDetector.visualize_detections(image, boxes, scores, classes, masks)
                    cv2.imwrite('%s/test_images/%s/%s/input/%004d.jpg' % (ROOT_DIR, DATASET, TYPE, image_id), image[:, :, ::-1])
                    cv2.imwrite('%s/test_images/%s/%s/output/%004d.jpg' % (ROOT_DIR, DATASET, TYPE, image_id), output_image[:, :, ::-1])
            except:
                break

    threads = []
    for ii in range(constants.PARALLEL_SIZE):
        obj_detector_thread_id, obj_detector_output_queue = object_detector.add_user()
        object_pipes = (obj_detector_thread_id, object_detector.input_queue, obj_detector_output_queue)
        if constants.USE_MULTIPROCESSING:
            thread = multiprocessing.Process(target=run_detector, args=(object_pipes,))
        else:
            thread = threading.Thread(target=run_detector, args=(object_pipes,))
        threads.append(thread)


    start_time = time.time()
    for thread in threads:
        thread.start()

    object_detector.create_net()
    object_detector.load_weights()
    sess.graph.finalize()

    for thread in threads:
        thread.join()

    total_time = time.time() - start_time

    print('total time %.3f' % total_time)
    print('per image time %.3f' % (total_time / NUM_IMAGES))

if __name__ == '__main__':
    main()
