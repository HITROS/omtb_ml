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

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String

from collections import defaultdict
import argparse
import cv2
import glob
import logging
import os
import sys
import time
########################################################################
import pycocotools.mask as mask_util
import numpy as np
from detectron.utils.colormap import colormap
###########################################################################
from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine

import detectron.utils.c2 as c2_utils
from category_name import get_coco_dataset
import detectron.utils.vis as vis_utils

from semantic_mapping.srv import MaskInfer

c2_utils.import_detectron_ops()
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '-cfg',
        dest='cfg',
        help='cfg model file',
        default=None,
        type=str
    )
    parser.add_argument(
        '-dir',
        dest='dir',
        help='dir file',
        default=None,
        type=str
    )
    parser.add_argument(
        '-wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '-thresh',
        dest='thresh',
        help='Threshold for visualizing detections',
        default=0.7,
        type=float
    )
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
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


class Infer:
    def __init__(self, args):
        self.logger = logging.getLogger(__name__)
        merge_cfg_from_file(args.cfg)
        cfg.NUM_GPUS = 1
        args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
        assert_and_infer_cfg(cache_urls=False)
        assert not cfg.MODEL.RPN_ONLY, \
            'RPN models are not supported'
        assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
            'Models that require precomputed proposals are not supported'
        self.model = infer_engine.initialize_model_from_cfg(args.weights)
        self.coco_dataset = get_coco_dataset()
        self.image_srv = rospy.Service('infer_image', MaskInfer, self.infer_image)
        self.thresh = args.thresh
        self.dir = args.dir

    def infer_image(self, req):
        # bridge = CvBridge()
        # image = bridge.imgmsg_to_cv2(req.rgb_image, desired_encoding='bgr8')
        path = self.dir + "/" + str(req.num)
        image = cv2.imread(path + "/image.png")
        category = []
        mask_image_all = []
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                self.model, image, None, timers=timers
            )
        self.logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            self.logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
        boxes, segms, keypoints, classes = vis_utils.convert_from_cls_format(cls_boxes, cls_segms, None)
        if segms is not None and len(segms) > 0:
            masks = mask_util.decode(segms)
        file_name = path + "/image.txt"
        file = open(file_name, 'w')
        file.write(str(0) + str(" "))
        file.close()
        ground_image = np.ones((480, 640, 1), np.uint8)
        class_num = 1
        for item in range(len(boxes)):
            score = boxes[item, -1]
            if score < self.thresh:
                continue
            idx = np.nonzero((masks[..., item]))
            mask_image = np.zeros((480, 640, 1), np.uint8)
            mask_image[idx[0], idx[1], :] = 255
            ground_image[idx[0], idx[1], :] = 0
            # mask_msg = bridge.cv2_to_imgmsg(mask_image, encoding='bgr8')
            # category.append(classes[item])
            # mask_image_all.append(mask_image)
            cv2.imwrite(path + "/" + str(classes[item]) + ".png", mask_image)
            file_name = path + "/image.txt"
            f = open(file_name, 'a')
            f.write(str(classes[item]) + str(" "))
            f.close()
            class_num += 1
        cv2.imwrite(path + "/0.png", ground_image)
        return class_num



def main(args):
    rospy.init_node("infer_image_server", anonymous = True)
    infer_server = Infer(args)
    rospy.spin()


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
