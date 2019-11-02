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
import rospy
from gazebo_msgs.srv import DeleteModel
from gazebo_msgs.msg import ModelStates


def callback(data):
    model_list = ['ground_plane', 'empty_room', 'camera']
    model_name = data.name
    for item in model_name:
        if item not in model_list:
            rospy.wait_for_service('gazebo/delete_model')
            del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
            del_model_prox(item)
    rospy.signal_shutdown("All other models have been cleared!!!")


def main():
    rospy.init_node("delete", anonymous=False)
    rospy.Subscriber("/gazebo/model_states", ModelStates, callback)
    rospy.spin()


if __name__ == '__main__':
    main()
