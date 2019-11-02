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

from __future__ import print_function
import sys
import rospy
import cv2
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import argparse
import math
from geometry_msgs.msg import Pose
from respawn import Respawn


def parse_args():
    parser = argparse.ArgumentParser(description='Save picture')
    parser.add_argument(
        '-dir',
        dest='dir',
        help='folder for saving picture (REQUIRED)',
        default=None,
        type=str)
    parser.add_argument(
        '-file',
        dest='file',
        help='file name (REQUIRED)',
        default=None,
        type=str)
    parser.add_argument(
        '-model',
        dest='model',
        help='model name (REQUIRED)',
        default=None,
        type=str)
    parser.add_argument(
        '-turns',
        dest='turns',
        help='turns (default = 1)',
        default=1,
        type=int)

    parser.add_argument(
        '-x',
        dest='x',
        help='x offset (default = 0)',
        default=0,
        type=float)
    parser.add_argument(
        '-y',
        dest='y',
        help='y offset (default = 0)',
        default=0,
        type=float)
    parser.add_argument(
        '-z',
        dest='z',
        help='z offset (default = 0)',
        default=0,
        type=float)
    parser.add_argument(
        '-R',
        dest='R',
        help='R rotation (default = 0)',
        default=0,
        type=float)
    parser.add_argument(
        '-P',
        dest='P',
        help='P rotation (default = 0)',
        default=0,
        type=float)
    parser.add_argument(
        '-Y',
        dest='Y',
        help='Y rotation (default = 0)',
        default=0,
        type=float)
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

    if (parser.parse_args().file is None) or \
       (parser.parse_args().model is None) or \
       (parser.parse_args().dir is None):
        parser.print_help()
        sys.exit(1)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def get_pose(object_position):
    object_pose = Pose()
    object_pose.position.x = object_position[0]
    object_pose.position.y = object_position[1]
    object_pose.position.z = object_position[2]
    roll = object_position[3]
    pitch = object_position[4]
    yaw = object_position[5]
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    object_pose.orientation.w = cy * cr * cp + sy * sr * sp
    object_pose.orientation.x = cy * sr * cp - sy * cr * sp
    object_pose.orientation.y = cy * cr * sp + sy * sr * cp
    object_pose.orientation.z = sy * cr * cp - cy * sr * sp
    return object_pose


class image_converter:
    def __init__(self, my_dir, my_env, model_name, turns):
        self.bridge = CvBridge()
        self.all_done = np.zeros(80)
        self.model_name = model_name
        self.my_dir = my_dir
        self.my_env = my_env
        self.turns = turns
        self.folder_img = self.my_dir + os.sep + 'img'
        self.folder_mask = self.my_dir + os.sep + 'mask'
        self.folder_img_model = self.my_dir + os.sep + 'img' + os.sep + self.model_name
        self.folder_mask_model = self.my_dir + os.sep + 'mask' + os.sep + self.model_name
        self.mkdir()
        for i in range(80):
            self.image_sub = rospy.Subscriber("/camera" + str(i+1) + "/image_raw", Image, self.callback, (i+1))

    def callback(self, data, num):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # (rows, cols, channels) = cv_image.shape
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        ground_lower = np.array([0, 0, 130])
        ground_upper = np.array([0, 0, 160])
        wall1_lower = np.array([0, 0, 90])
        wall1_upper = np.array([0, 0, 120])
        wall2_lower = np.array([0, 0, 170])
        wall2_upper = np.array([0, 0, 200])
        mask_ground = cv2.inRange(hsv, ground_lower, ground_upper)
        mask_wall = cv2.bitwise_or(cv2.inRange(hsv, wall1_lower, wall1_upper),
                                   cv2.inRange(hsv, wall2_lower, wall2_upper))
        mask = cv2.bitwise_or(mask_ground, mask_wall)
        kernel = np.ones((3, 3), np.int8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_output = cv2.bitwise_not(mask)
        # img_mask_final = np.zeros([480, 640, 1], dtype=np.uint8)
        # height = img_mask_final.shape[0]
        # width = img_mask_final.shape[1]
        # channels = img_mask_final.shape[2]
        # for row in range(height):
        #     for col in range(width):
        #             img_mask_final[row][col][0] = mask_output[row][col]
        cv2.imwrite(self.folder_img_model + os.sep + str(num + (self.turns - 1) * 80) + '.png', cv_image)
        # cv2.imwrite(self.folder_mask_model + os.sep + str(num+(self.turns-1)*80)+'.png', img_mask_final)
        cv2.imwrite(self.folder_mask_model + os.sep + str(num + (self.turns - 1) * 80) + '.png', mask_output)
        if not np.sum(self.all_done == 0):
            rospy.signal_shutdown("All done!!!")
        self.all_done[num - 1] = 1

    def mkdir(self):
        exist_folder = os.path.exists(self.my_dir)
        if not exist_folder:
            os.makedirs(self.my_dir)
        exist_folder_img = os.path.exists(self.folder_img)
        exist_folder_mask = os.path.exists(self.folder_mask)
        if not exist_folder_img:
            os.makedirs(self.folder_img)
        if not exist_folder_mask:
            os.makedirs(self.folder_mask)
        exist_folder_img_model = os.path.exists(self.folder_img_model)
        exist_folder_mask_model = os.path.exists(self.folder_mask_model)
        if not exist_folder_img_model:
            os.makedirs(self.folder_img_model)
        if not exist_folder_mask_model:
            os.makedirs(self.folder_mask_model)


def main(args):
    rospy.init_node("save_piv", anonymous=False)
    pose = get_pose([args.x, args.y, args.z, args.R, args.P, args.Y])
    my_env = Respawn(model_path=args.file,
                     model_pose=pose, model_name=args.model)
    my_env.respawnModel()
    my_dir = args.dir
    model_name = args.model
    turns = args.turns
    rospy.sleep(1.)
    try:
        image_converter(my_dir, my_env, model_name, turns)
    except rospy.ROSInterruptException:
        pass
    rospy.spin()


if __name__ == '__main__':
    args = parse_args()
    main(args)
