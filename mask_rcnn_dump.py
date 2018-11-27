import pdb
import time
import os
import cv2
import numpy as np
import threading
import h5py
import random
from utils import py_util
from utils import game_util

from graph import graph_obj

import constants
from my_utils.util import drawing
from darknet_object_detection import detector
from EmbodiedQA import house3dEnv

DEBUG = False
MAX_OBJECTS_PER_IMAGE = 100
TYPE = 'train'

USE_HDF5 = False

SCREEN_WIDTH = 448
SCREEN_HEIGHT = 448

if not constants.EQA:
    if TYPE == 'val_seen':
        MAX_NUM_IMAGES = 100
        scene_numbers = list(range(1, 6))
        PARALLEL_SIZE = 1
        MAX_IMAGES_PER_SCENE = 5
    elif TYPE == 'val_unseen':
        scene_numbers = list(range(6, 31))
        MAX_NUM_IMAGES = 10 * len(scene_numbers)
        PARALLEL_SIZE = 1
        MAX_IMAGES_PER_SCENE = 2
    else:
        MAX_NUM_IMAGES = 100000
        scene_numbers = list(range(6, 31))
        PARALLEL_SIZE = 16
        MAX_IMAGES_PER_SCENE = 50
else:
    if TYPE == 'val_unseen':
        MAX_NUM_IMAGES = 1000
        scene_names = constants.ALL_ROOMS['val']
        PARALLEL_SIZE = 16
    elif TYPE == 'val_seen':
        scene_names = constants.ALL_ROOMS['train']
        MAX_NUM_IMAGES = 10 * len(scene_names)
        PARALLEL_SIZE = 16
    else:
        MAX_NUM_IMAGES = 100000
        scene_names = constants.ALL_ROOMS['train']
        PARALLEL_SIZE = 16
    MAX_IMAGES_PER_SCENE = int(np.ceil(MAX_NUM_IMAGES / len(scene_names)))


if constants.EQA:
    path_var = 'eqa'
else:
    path_var = 'thor'
base_dir = 'Mask_RCNN/data/%s/%s/' % (path_var, TYPE)
type_str = TYPE + '_'

if not os.path.exists(base_dir):
    os.makedirs(base_dir)
time_str = py_util.get_time_str()
if USE_HDF5:
    dataset = h5py.File(base_dir + 'data_' + type_str + time_str + '.h5', 'w')
    dataset.create_dataset('images/color_images', (MAX_NUM_IMAGES, SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
    dataset.create_dataset('images/seg_images', (MAX_NUM_IMAGES, SCREEN_HEIGHT, SCREEN_WIDTH),
                           dtype=np.uint8)
    dataset.create_dataset('images/class_labels', (MAX_NUM_IMAGES, MAX_OBJECTS_PER_IMAGE), dtype=np.uint8)
    dataset.create_dataset('images/num_objects', (MAX_NUM_IMAGES, 1), dtype=np.uint8)
else:
    dataset = {}
    dataset['images/class_labels'] = np.zeros((MAX_NUM_IMAGES, MAX_OBJECTS_PER_IMAGE), dtype=np.uint8)
    dataset['images/num_objects'] = np.zeros((MAX_NUM_IMAGES, 1), dtype=np.uint8)
    dataset['images/angle'] = np.zeros((MAX_NUM_IMAGES, 1), dtype=np.int16)
    if not os.path.exists(base_dir + 'color_images'):
        os.makedirs(base_dir + 'color_images')
    if not os.path.exists(base_dir + 'seg_images'):
        os.makedirs(base_dir + 'seg_images')


global_image_count = 0
data_lock = threading.Lock()
start_time = time.time()

edge_box = np.array([0, 1, SCREEN_WIDTH, SCREEN_WIDTH - 1])

def make_images(thread_ind):
    global global_image_count
    global start_time
    if constants.EQA:
        render_types = {'color', 'instance_segmentation', 'object_segmentation'}
        if DEBUG:
            render_types.add('debug')
        env = house3dEnv.MultiprocessEnv(scene_width=SCREEN_WIDTH, scene_height=SCREEN_HEIGHT,
                                         render_types=render_types)
    else:
        env = game_util.create_env(player_screen_width=SCREEN_WIDTH, player_screen_height=SCREEN_HEIGHT)
    try:
        while global_image_count < MAX_NUM_IMAGES:
            if not constants.EQA:
                scene_num = random.choice(scene_numbers)
                scene_name = 'FloorPlan%d' % scene_num
                event = game_util.reset(env, scene_name, render_depth_image=False, render_class_image=False, render_object_image=True)
                event = env.random_initialize(randomize_open=True, remove_prob=0, max_num_repeats=10)
                grid_file = 'layouts/high_res/%s-layout.npy' % scene_name
                graph = graph_obj.Graph(grid_file, use_gt=True)
                graph_points = graph.points
            else:
                scene_name = random.choice(scene_names)
                scene_num = constants.ROOM_NAME_TO_IND[scene_name]
                event = game_util.reset(env, scene_name)
                graph_points = env.points

            agent_height = event.metadata['agent']['position']['y']
            used_locations = set()
            for ii in range(MAX_IMAGES_PER_SCENE):
                start_point = None
                while start_point is None:
                    start_point = random.randint(0, graph_points.shape[0] - 1)
                    start_point = graph_points[start_point,:].copy()
                    start_point = (start_point[0], start_point[1], random.randint(0, 3), random.randint(-1, 2))
                    if start_point in used_locations:
                        start_point = None
                        continue
                    elif constants.EQA and ii % 10 != 0:
                        # Limit the number of outdoor images because they are very similar and not very useful.
                        room_type = env.get_room_type(start_point[1] * constants.AGENT_STEP_SIZE,
                                                      start_point[0] * constants.AGENT_STEP_SIZE)
                        if 'outdoor' in room_type:
                            start_point = None
                            continue
                used_locations.add(start_point)
                action = {'action': 'TeleportFull',
                        'x': start_point[0] * constants.AGENT_STEP_SIZE,
                        'y': agent_height,
                        'z': start_point[1] * constants.AGENT_STEP_SIZE,
                        'rotateOnTeleport': True,
                        'rotation': int(start_point[2] * 90),
                        'horizon': int(start_point[3] * 30),
                        }
                event = env.step(action)

                seg_image = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8)
                if DEBUG:
                    bad_mask = np.zeros_like(seg_image)
                ii = 1
                class_labels = np.zeros((MAX_OBJECTS_PER_IMAGE), dtype=np.uint8)
                num_objects = 0
                for (key, mask) in event.instance_masks.items():
                    if '|' not in key:
                        continue
                    cls = key.split('|')[0]
                    if cls not in constants.OBJECTS_SET:
                        continue
                    bbox = event.instance_detections2D[key]
                    if np.min(bbox[2:] - bbox[:2]) < 5:
                        if DEBUG:
                            print('object too small')
                            bad_mask[mask] = 4
                        continue
                    if len(np.intersect1d(bbox, edge_box)) > 0:
                        if len(np.intersect1d(bbox[[0, 2]], edge_box)) > 0:
                            # Width
                            cropped_length = bbox[2] - bbox[0]
                        else:
                            # Height
                            cropped_length = bbox[3] - bbox[1]

                        # on border
                        if cropped_length < SCREEN_WIDTH / 20:
                            if DEBUG:
                                print('probably cropped out')
                                bad_mask[mask] = 3
                            continue

                    label_ind = constants.OBJECT_CLASS_TO_ID[cls]
                    if cls == 'Cabinet':
                        row_count = np.sum(mask, axis=0)
                        col_count = np.sum(mask, axis=1)

                        row_counts_masked = row_count[row_count > 0]
                        col_counts_masked = col_count[col_count > 0]

                        row_exist_count = len(row_counts_masked)
                        col_exist_count = len(col_counts_masked)
                        min_row_col = min(row_exist_count, col_exist_count)
                        max_row_col = max(row_exist_count, col_exist_count)
                        '''
                        if max_row_col / min_row_col > 3:
                            # Ignore cabinets that are 3x as tall as wide or more, they are probably open
                            bad_mask[mask] = 2
                            print('tall cabinet, ignore')
                            continue
                        '''
                        row_sum_max = np.max(row_counts_masked)
                        col_sum_max = np.max(col_counts_masked)
                        if (len(row_counts_masked) / row_sum_max > 6 or
                            len(col_counts_masked) / col_sum_max > 6):
                            if DEBUG:
                                bad_mask[mask] = 1
                                print('thin cabinet, ignore')
                                print(len(row_counts_masked) / row_sum_max)
                                print(len(col_counts_masked) / col_sum_max)
                            continue


                    seg_image[mask] = ii
                    class_labels[ii] = label_ind
                    num_objects += 1
                    ii += 1
                    if ii >= MAX_OBJECTS_PER_IMAGE:
                        break

                data_lock.acquire()

                if DEBUG:
                    cv2.imshow('image', event.frame[:, :, ::-1])
                    seg_image[0, 0] = constants.NUM_CLASSES
                    cv2.imshow('seg', (seg_image * 255.0 / num_objects).astype(np.uint8))
                    cv2.imshow('bad_mask', (bad_mask * 255.0 / np.max(bad_mask)).astype(np.uint8))
                    cv2.imshow('debug', event.debug_frame[:, :, ::-1])
                    cv2.waitKey(0)

                if global_image_count >= MAX_NUM_IMAGES:
                    data_lock.release()
                    break
                local_image_count = global_image_count
                global_image_count += 1
                if global_image_count % 100 == 0:
                    print('Num images', global_image_count)
                    print('time %.3f' % ((time.time() - start_time) / 100))
                    start_time = time.time()
                    if global_image_count % 1000 == 0:
                        np.save(base_dir + 'class_labels.npy', dataset['images/class_labels'])
                        np.save(base_dir + 'num_objects.npy', dataset['images/num_objects'])
                        np.save(base_dir + 'angle.npy', dataset['images/angle'])

                if USE_HDF5:
                    dataset['images/color_images'][local_image_count, ...] = event.frame
                    dataset['images/seg_images'][local_image_count, ...] = seg_image
                    dataset['images/class_labels'][local_image_count, ...] = class_labels
                    dataset['images/num_objects'][local_image_count, ...] = num_objects
                    dataset.flush()
                    data_lock.release()
                else:
                    data_lock.release()
                    dataset['images/class_labels'][local_image_count, ...] = class_labels
                    dataset['images/num_objects'][local_image_count, ...] = num_objects
                    dataset['images/angle'][local_image_count, ...] = int(event.pose[3] / 1000)
                    #scipy.misc.imsave(base_dir + 'color_images/%08d.jpg' % local_image_count, event.frame)
                    cv2.imwrite(base_dir + 'color_images/%08d.jpg' % local_image_count, event.frame[:, :, ::-1])
                    #cv2.imwrite(base_dir + 'color_images/%08d_3.jpg' % local_image_count, event.frame[:, :, ::-1], (cv2.IMWRITE_JPEG_QUALITY, 100))
                    #cv2.imwrite(base_dir + 'color_images/%08d_4.png' % local_image_count, event.frame[:, :, ::-1])
                    cv2.imwrite(base_dir + 'seg_images/%08d.png' % local_image_count, seg_image)


    finally:
        try:
            print('stopping')
            np.save(base_dir + 'class_labels.npy', dataset['images/class_labels'])
            np.save(base_dir + 'num_objects.npy', dataset['images/num_objects'])
            np.save(base_dir + 'angle.npy', dataset['images/angle'])
            '''
            x = np.concatenate(points[0])
            y = np.concatenate(points[1])
            import scipy.stats
            print(scipy.stats.linregress(x, y))
            pdb.set_trace()
            #env.stop()
            print('stopped')
            '''
        except:
            pdb.set_trace()
            print('exception')
            np.save(base_dir + 'class_labels.npy', dataset['images/class_labels'])
            np.save(base_dir + 'num_objects.npy', dataset['images/num_objects'])
            np.save(base_dir + 'angle.npy', dataset['images/angle'])

        print('done')
        env.stop()


if DEBUG:
    make_images(0)
else:
    threads = [threading.Thread(target=make_images, args=(ii,)) for ii in range(PARALLEL_SIZE)]
    for thread in threads:
        thread.start()
        time.sleep(1)
    for thread in threads:
        thread.join()

print('bottom done')
