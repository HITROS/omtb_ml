/*******************************************************************************
* Copyright 2019 HITROS CO., LTD.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/* Author: Yu Fu */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <boost/timer.hpp>

#include <ros/ros.h>
#include <std_msgs/Int64.h>
#include <std_msgs/Int64MultiArray.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Transform.h>
#include <tf/transform_broadcaster.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_kdl.h>

// opencv 用于图像数据读取与处理
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// 使用Eigen的Geometry模块处理3d运动
#include <Eigen/Eigen>

// boost.format 字符串处理
//#include <boost/format.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/console/parse.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "semantic_pick/MaskInfer.h"
#include "semantic_pick/ObjectPosition.h"

#define CLASS_UPPER_LIMIT 7
#define CAMERA_NUM 8


typedef message_filters::sync_policies::ApproximateTime
<sensor_msgs::PointCloud2,
sensor_msgs::PointCloud2,
sensor_msgs::PointCloud2,
sensor_msgs::PointCloud2,
sensor_msgs::PointCloud2,
sensor_msgs::PointCloud2,
sensor_msgs::PointCloud2,
sensor_msgs::PointCloud2>
MySyncPolicy;

using namespace std;
using namespace cv;
typedef pcl::PointXYZRGBA PointAT;
typedef pcl::PointCloud<PointAT> PointACloudT;
bool isFirst = true;
bool isReconstruction = false;
long my_num = 0;
std::stringstream ss;
std::string file_name;
std::string text_name;
std::string image_name;
std::string string_name;
std::string file_name_pcd;

int v1(0); //定义两个窗口v1，v2，窗口v1用来显示初始位置，v2用以显示配准过程
int v2(0);

PointACloudT object_visiable[CLASS_UPPER_LIMIT];
PointACloudT objects[CLASS_UPPER_LIMIT];
PointACloudT object_ground;
PointACloudT cloud_all;
Eigen::Vector4f model_centroid[CLASS_UPPER_LIMIT];

int rm_dir(std::string dir_full_path) {
    DIR* dirp = opendir(dir_full_path.c_str());
    if(!dirp)
    {
        return -1;
    }
    struct dirent *dir;
    struct stat st;
    while((dir = readdir(dirp)) != NULL)
    {
        if(strcmp(dir->d_name,".") == 0
           || strcmp(dir->d_name,"..") == 0)
        {
            continue;
        }
        std::string sub_path = dir_full_path + '/' + dir->d_name;
        if(lstat(sub_path.c_str(),&st) == -1)
        {
            //Log("rm_dir:lstat ",sub_path," error");
            continue;
        }
        if(S_ISDIR(st.st_mode))
        {
            if(rm_dir(sub_path) == -1) // 如果是目录文件，递归删除
            {
                closedir(dirp);
                return -1;
            }
            rmdir(sub_path.c_str());
        }
        else if(S_ISREG(st.st_mode))
        {
            unlink(sub_path.c_str());     // 如果是普通文件，则unlink
        }
        else
        {
            //Log("rm_dir:st_mode ",sub_path," error");
            continue;
        }
    }
    if(rmdir(dir_full_path.c_str()) == -1)//delete dir itself.
    {
        closedir(dirp);
        return -1;
    }
    closedir(dirp);
    return 0;
}

//Remove files or dirs
bool Remove(std::string file_name) {
    std::string file_path = file_name;
    struct stat st;
    if (lstat(file_path.c_str(),&st) == -1) {
        return EXIT_FAILURE;
    }
    if (S_ISREG(st.st_mode)) {
        if (unlink(file_path.c_str()) == -1) {
            return EXIT_FAILURE;
        }
    }
    else if(S_ISDIR(st.st_mode)) {
        if(file_name == "." || file_name == "..") {
            return EXIT_FAILURE;
        }
        if(rm_dir(file_path) == -1)//delete all the files in dir.
        {
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}


void callback(const sensor_msgs::PointCloud2ConstPtr& camera1,
              const sensor_msgs::PointCloud2ConstPtr& camera2,
              const sensor_msgs::PointCloud2ConstPtr& camera3,
              const sensor_msgs::PointCloud2ConstPtr& camera4,
              const sensor_msgs::PointCloud2ConstPtr& camera5,
              const sensor_msgs::PointCloud2ConstPtr& camera6,
              const sensor_msgs::PointCloud2ConstPtr& camera7,
              const sensor_msgs::PointCloud2ConstPtr& camera8,
              ros::NodeHandle nh
              )
{
  if(isFirst)
  {
    isFirst = false;
  }
  else if(!isFirst && !isReconstruction)
  {
    sensor_msgs::PointCloud2 input_cloud[CAMERA_NUM];
    input_cloud[0] = *camera1;
    input_cloud[1] = *camera2;
    input_cloud[2] = *camera3;
    input_cloud[3] = *camera4;
    input_cloud[4] = *camera5;
    input_cloud[5] = *camera6;
    input_cloud[6] = *camera7;
    input_cloud[7] = *camera8;
    for(int camera_iter=4;camera_iter<CAMERA_NUM;camera_iter++)
    {
      ss.clear();
      ss.str("");
      ss << "/tmp/infer/" << camera_iter;
      ss >> file_name;
      if (access(file_name.c_str(), 0) == -1)//返回值为-1，表示不存在
      {
        cout<<file_name<<endl;
        mkdir(file_name.c_str(), S_IRWXU|S_IRWXG|S_IRWXO);
      }
      sensor_msgs::Image image;
      static tf::TransformListener listener;
      tf::StampedTransform transform;
      try
      {
        string tf_name;
        ss.clear();
        ss.str("");
        ss<< "multi_camera/camera" << (camera_iter+1) << "_optical_frame";
        ss >> tf_name;
        listener.waitForTransform("world", tf_name, ros::Time(0), ros::Duration(1.0));
        listener.lookupTransform("world", tf_name, ros::Time(0), transform);
      }
      catch (tf::TransformException ex)
      {
        return;
        ROS_ERROR("%s",ex.what());
        ros::Duration(1.0).sleep();
      }
      Eigen::Quaterniond q(transform.getRotation().getW(), transform.getRotation().getX(), transform.getRotation().getY(), transform.getRotation().getZ());
      Eigen::Isometry3d t(q);
      Eigen::Matrix4d matrix_now;
      t(0,3) = transform.getOrigin().x();
      t(1,3) = transform.getOrigin().y();
      t(2,3) = transform.getOrigin().z();
      matrix_now = t.matrix();
      pcl::PointCloud<pcl::PointXYZRGBA> cloud;
      pcl::fromROSMsg(input_cloud[camera_iter], cloud);
      pcl::toROSMsg(input_cloud[camera_iter], image);
      cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8);
      imwrite(file_name + "/image.png", cv_ptr->image);
      ros::ServiceClient client = nh.serviceClient<semantic_pick::MaskInfer>("infer_image");
      semantic_pick::MaskInfer srv;
      srv.request.num = camera_iter;
      ss.clear();
      ss.str("");
      ss<<file_name << "/my.pcd";
      ss>>file_name_pcd;
      pcl::io::savePCDFile(file_name_pcd, cloud);

      if (client.call(srv))
      {

        int class_num = srv.response.class_num;
        int class_data[CLASS_UPPER_LIMIT];

        // Load mask
        ss.clear();
        ss.str("");
        ss << file_name << "/image.txt";
        ss >> text_name;
        ifstream text_file;
        text_file.open(text_name);//打开文件
        int num = 0;
        while(text_file>>class_data[num]){num++;}
        text_file.close();
        pcl::transformPointCloud( cloud, cloud, matrix_now);
        for (int i=0;i<class_num;i++)
        {
          PointACloudT objects_temp;
          std::vector<int> indexs;
          // Load mask
          ss.clear();
          ss.str("");
          ss << file_name << "/" << class_data[i]<<".png";
          ss >> image_name;
          Mat cv_image = imread(image_name, 0);
          for(int i_row=0;i_row<cv_image.rows;i_row++)
          {
              for(int i_col=0;i_col<cv_image.cols;i_col++)
              {
                  if(cv_image.at<uchar>(i_row,i_col) != 0 &&
                      isfinite(cloud.points[i_row*cv_image.cols + i_col].x)  &&
                      isfinite(cloud.points[i_row*cv_image.cols + i_col].y)  &&
                      isfinite(cloud.points[i_row*cv_image.cols + i_col].z)
                      )
                    indexs.push_back(i_row*cv_image.cols + i_col);
              }
          }
          pcl::copyPointCloud(cloud, indexs, objects_temp);
          if (objects[class_data[i]].points.size()==0)
            objects[class_data[i]] = objects_temp;
          else
            objects[class_data[i]] += objects_temp;
          ss.clear();
          ss.str("");
          ss<<file_name << "/" << class_data[i]<<".pcd";
          ss>>file_name_pcd;
          pcl::io::savePCDFile(file_name_pcd, objects[class_data[i]]);
        }
      }
      else
      {
        ROS_ERROR("Failed to call service add_two_ints");
      }
    }
    isReconstruction = true;
    for (int i=1;i<CLASS_UPPER_LIMIT;i++)
    {
      if (objects[i].points.size()!=0)
      {
        pcl::compute3DCentroid(objects[i], model_centroid[i]);
      }
    }
  }
  else
  {

  }
}

bool service_callback(semantic_pick::ObjectPositionRequest& request, semantic_pick::ObjectPositionResponse& response)
{
  if (isReconstruction)
  {
    for (int i=1;i<CLASS_UPPER_LIMIT;i++)
    {
      if (objects[i].points.size()!=0)
      {
        response.object_class.push_back(i);
        response.position_x.push_back(model_centroid[i][0]);
        response.position_y.push_back(model_centroid[i][1]);
        response.position_z.push_back(model_centroid[i][2]);
      }
    }
    return true;
  }
  else
  {
    return false;
  }
}


int main(int argc, char **argv)
{
  string path ="/tmp/infer";
  if (access(path.c_str(), 0) == -1)//返回值为-1，表示不存在
  {
    mkdir(path.c_str(), S_IRWXU|S_IRWXG|S_IRWXO);
  }
  else
  {
    Remove(path);
    mkdir(path.c_str(), S_IRWXU|S_IRWXG|S_IRWXO);
  }
  ros::init(argc, argv, "mapping");
  ros::NodeHandle nh;
  message_filters::Subscriber<sensor_msgs::PointCloud2> camera1(nh,"/multi_camera/camera1/depth/points", 1);
  message_filters::Subscriber<sensor_msgs::PointCloud2> camera2(nh,"/multi_camera/camera2/depth/points", 1);
  message_filters::Subscriber<sensor_msgs::PointCloud2> camera3(nh,"/multi_camera/camera3/depth/points", 1);
  message_filters::Subscriber<sensor_msgs::PointCloud2> camera4(nh,"/multi_camera/camera4/depth/points", 1);
  message_filters::Subscriber<sensor_msgs::PointCloud2> camera5(nh,"/multi_camera/camera5/depth/points", 1);
  message_filters::Subscriber<sensor_msgs::PointCloud2> camera6(nh,"/multi_camera/camera6/depth/points", 1);
  message_filters::Subscriber<sensor_msgs::PointCloud2> camera7(nh,"/multi_camera/camera7/depth/points", 1);
  message_filters::Subscriber<sensor_msgs::PointCloud2> camera8(nh,"/multi_camera/camera8/depth/points", 1);

  // ExactTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
  message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(50), camera1, camera2, camera3, camera4, camera5, camera6, camera7, camera8);
//    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), image_sub, depth_sub);
  sync.registerCallback(boost::bind(&callback, _1, _2, _3, _4, _5, _6, _7, _8, nh));
  ros::ServiceServer service = nh.advertiseService("provide_objects", service_callback);

  ros::spin();

  return 0;
}
