#!/usr/bin/env python

##############################################################################
# Copyright 2019 HITROS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

# Author: Yu Fu

import re
import sys
import os
import shutil
import cv2
import numpy as np
import random
import argparse
import datetime
from PIL import Image
from pycocotools import mask
from skimage import measure
from itertools import groupby
import json
import fnmatch


INFO = {
    "description": "Dataset",
    "url": "https://github.com/HITROS/omtb_ml",
    "version": "0.1.0",
    "year": 2019,
    "contributor": "Yu Fu",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "License",
        "url": "http://www.apache.org/licenses/LICENSE-2.0"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'bookshelf',
        'supercategory': 'gazebo',
    },
    {
        'id': 2,
        'name': 'cabinet',
        'supercategory': 'gazebo',
    },
    {
        'id': 3,
        'name': 'cafe_table',
        'supercategory': 'gazebo',
    },
    {
        'id': 4,
        'name': 'cardboard_box',
        'supercategory': 'gazebo',
    },
    {
        'id': 5,
        'name': 'car_wheel',
        'supercategory': 'gazebo',
    },
    {
        'id': 6,
        'name': 'cinder_block',
        'supercategory': 'gazebo',
    },
    {
        'id': 7,
        'name': 'coke_can',
        'supercategory': 'gazebo',
    },
    {
        'id': 8,
        'name': 'construction_barrel',
        'supercategory': 'gazebo',
    },
    {
        'id': 9,
        'name': 'construction_cone',
        'supercategory': 'gazebo',
    },
    {
        'id': 10,
        'name': 'drc_practice_blue_cylinder',
        'supercategory': 'gazebo',
    },
    {
        'id': 11,
        'name': 'drc_practice_hinged_door',
        'supercategory': 'gazebo',
    },
    {
        'id': 12,
        'name': 'ycb_banana',
        'supercategory': 'gazebo',
    },
    {
        'id': 13,
        'name': 'ycb_potted_meat_can',
        'supercategory': 'gazebo',
    },
]



def parse_args():
    parser = argparse.ArgumentParser(description='Save picture')
    parser.add_argument(
        '-dir',
        dest='dir',
        help='folder for saving picture (REQUIRED)',
        default=None,
        type=str)
    parser.add_argument(
        'name',
        help='name',
        default='',
        type=str)
    parser.add_argument(
        'log',
        help='log',
        default='',
        type=str)
    if parser.parse_args().dir is None:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def create_image_info(image_id, file_name, image_size,
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):
    image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[0],
            "height": image_size[1],
            "date_captured": date_captured,
            "license": license_id,
            "coco_url": coco_url,
            "flickr_url": flickr_url
    }
    return image_info


def filter_for_png(root, files):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    return files


def filter_for_annotations(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]
    return files


def resize_binary_mask(array, new_size):
    image = Image.fromarray(array.astype(np.uint8)*255)
    image = image.resize(new_size)
    return np.asarray(image).astype(np.bool_)


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation

    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.

    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)
    return polygons


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
                counts.append(0)
        counts.append(len(list(elements)))
    return rle


def create_annotation_info(annotation_id, image_id, category_info, binary_mask,
                           image_size=None, tolerance=2, bounding_box=None):
    if image_size is not None:
        binary_mask = resize_binary_mask(binary_mask, image_size)
    binary_mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    area = mask.area(binary_mask_encoded)
    if area < 1:
        return None
    if bounding_box is None:
        bounding_box = mask.toBbox(binary_mask_encoded)
    if category_info["is_crowd"]:
        is_crowd = 1
        segmentation = binary_mask_to_rle(binary_mask)
    else:
        is_crowd = 0
        segmentation = binary_mask_to_polygon(binary_mask, tolerance)
        if not segmentation:
            return None

    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_info["id"],
        "iscrowd": is_crowd,
        "area": area.tolist(),
        "bbox": bounding_box.tolist(),
        "segmentation": segmentation,
        "width": binary_mask.shape[1],
        "height": binary_mask.shape[0],
    }
    return annotation_info


def create(my_dir, dir_mask, dir_img, json_name):
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }
    image_id = 1
    segmentation_id = 1

    for _, _, maskfiles in os.walk(dir_mask):
        # go through each image
        for mask_filename in maskfiles:
            image_filename = mask_filename.split('_')[0] + '.png'
            # a = os.path.join(dir_img, image_filename)
            image = Image.open(os.path.join(dir_mask, mask_filename))
            # image.save(r'/home/fy/tmp/test1.png')
            image_info = create_image_info(
                image_id, os.path.basename(image_filename), image.size)
            coco_output["images"].append(image_info)
            class_id = [x['id'] for x in CATEGORIES if x['name'] in mask_filename][0]
            category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
            binary_mask = np.asarray(Image.open(os.path.join(dir_mask, mask_filename))
                                     .convert('L')).astype(np.uint8)
            # binary_mask_1 = np.asarray(Image.open(annotation_filename).convert('L')).astype(np.uint8)

            annotation_info = create_annotation_info(
                segmentation_id, image_id, category_info, binary_mask,
                image.size, tolerance=2)

            if annotation_info is not None:
                coco_output["annotations"].append(annotation_info)

            segmentation_id = segmentation_id + 1
            image_id = image_id + 1

    savename = '{}/'+json_name+'.json'
    with open(savename.format(my_dir), 'w') as output_json_file:
        json.dump(coco_output, output_json_file, indent=4)

def main(arg):
    my_dir = arg.dir
    imgDir = my_dir+r'/img'
    maskDir = my_dir+r'/mask'
    trainDir = my_dir+r'/train'
    valDir = my_dir+r'/val'
    trainDir_image = my_dir+r'/train/image'
    trainDir_mask = my_dir+r'/train/mask'
    valDir_image = my_dir + r'/val/image'
    valDir_mask = my_dir+r'/val/mask'
    exist_trainDir = os.path.exists(trainDir)
    exist_valDir = os.path.exists(valDir)
    if not exist_trainDir:
        os.makedirs(trainDir)
    if not exist_valDir:
        os.makedirs(valDir)
    exist_trainDir_image = os.path.exists(trainDir_image)
    exist_trainDir_mask = os.path.exists(trainDir_mask)
    if not exist_trainDir_image:
        os.makedirs(trainDir_image)
    if not exist_trainDir_mask:
        os.makedirs(trainDir_mask)
    exist_valDir_image = os.path.exists(valDir_image)
    exist_valDir_mask = os.path.exists(valDir_mask)
    if not exist_valDir_image:
        os.makedirs(valDir_image)
    if not exist_valDir_mask:
        os.makedirs(valDir_mask)

    imgnum = 1
    for parent, dirnames, _ in os.walk(imgDir):
        for dirname in dirnames:
            classnum = dirname
            # print dirname
            print classnum
            dirImage = parent + os.sep + dirname
            dirMask = maskDir + os.sep + classnum
            for _, _, filenames in os.walk(dirImage):
                for filename in filenames:
                    if filename.endswith('.png'):
                        imageName = str(imgnum) + '.png'
                        maskName = str(imgnum) + '_' + classnum + '.png'
                        if random.uniform(0, 1) < 0.75:
                            shutil.copyfile(os.path.join(dirImage, filename), os.path.join(trainDir_image, imageName))
                            shutil.copyfile(os.path.join(dirMask, filename), os.path.join(trainDir_mask, maskName))
                        else:
                            shutil.copyfile(os.path.join(dirImage, filename), os.path.join(valDir_image, imageName))
                            shutil.copyfile(os.path.join(dirMask, filename), os.path.join(valDir_mask, maskName))
                        imgnum += 1

    create(my_dir, trainDir_mask, trainDir_image, 'train')
    create(my_dir, valDir_mask, valDir_image, 'val')
    print 'all done!!!'


if __name__ == '__main__':
    arg = parse_args()
    main(arg)
