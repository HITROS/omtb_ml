################################################################################
# Set minimum required version of cmake, project name and compile options
################################################################################
cmake_minimum_required(VERSION 2.8.3)
project(semantic_mapping)

add_compile_options(-std=c++11)
################################################################################
# Find catkin packages and libraries for catkin and system dependencies
################################################################################
#find_package(catkin REQUIRED COMPONENTS roscpp rospy sensor_msgs std_msgs message_generation message_runtime cv_bridge)
#find_package(PCL REQUIRED COMPONENTS)
#include_directories(${PCL_INCLUDE_DIRS})
#link_directories(${PCL_LIBRARY_DIRS})
#find_package(OpenCV REQUIRED)

find_package(PCL REQUIRED COMPONENTS)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
find_package(pcl_conversions)
find_package(PCL REQUIRED COMPONENTS)
find_package(OpenCV REQUIRED)
find_package(catkin REQUIRED COMPONENTS
    roscpp
    rosmsg
    rospy
    std_msgs
    message_generation
    message_runtime
    sensor_msgs
    geometry_msgs
    cv_bridge
    tf

)
################################################################################
# Setup for python modules and scripts
################################################################################
catkin_python_setup()

################################################################################
# Declare ROS messages, services and actions
################################################################################
add_service_files(
  DIRECTORY srv
  FILES MaskInfer.srv
)

generate_messages(
  DEPENDENCIES sensor_msgs
)
################################################################################
# Declare ROS dynamic reconfigure parameters
################################################################################

################################################################################
# Declare catkin specific configuration to be passed to dependent projects
################################################################################
catkin_package(
  CATKIN_DEPENDS roscpp rospy rosmsg sensor_msgs  geometry_msgs std_msgs message_generation message_runtime cv_bridge tf)

################################################################################
# Build
################################################################################
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

link_directories(
  ${PCL_LIBRARY_DIRS}
)

add_executable(mapping
  src/mapping.cpp
)

target_link_libraries(mapping
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES})

################################################################################
# Install
################################################################################
catkin_install_python(PROGRAMS
  nodes/infer.py
  nodes/category_name.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY nodes srv
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
################################################################################
# Test
################################################################################
