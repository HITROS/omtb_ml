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

#include "semantic_mapping/MaskInfer.h"

#define CLASS_UPPER_LIMIT 20

using namespace std;
using namespace cv;
typedef pcl::PointXYZRGBA PointAT;
typedef pcl::PointCloud<PointAT> PointACloudT;
bool isFirst = true;
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

int bgr_list[][3] =
{
  {120, 120, 120},      //灰 0
  {0, 0, 255},    //红 1
  {0, 255,  0},   //绿 2
  {255, 0, 0},    //蓝 3
  {0, 255, 255},  //黄 4
  {255, 0, 255},  //洋红 5
  {255, 255, 0},  //青 6
  {0, 0, 128},    //栗色 7
  {0, 128,  0},   //纯绿 8
  {128, 0, 0},    //深蓝 9
  {0, 128, 128},  //橄榄 10
  {128, 0, 128},  //紫色 11
  {128, 128, 0},  //深青色 12
  {0, 0, 64},     //暗红 13
  {0, 64, 0},     //暗绿 14
  {64, 0, 0},     //暗蓝 15
  {0, 64, 64},    //暗黄 16
  {64, 0, 64},    //暗紫 17
  {64, 64, 0},    //暗青 18
};

void simpleVis (boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer, char* cam_file)
{
    viewer->loadCameraParameters(cam_file);
    viewer->createViewPort(0.0,0.0,0.5,1.0,v1);  //四个窗口参数分别对应x_min,y_min,x_max.y_max.
    viewer->createViewPort(0.5,0.0,1.0,1.0,v2);
    viewer->setBackgroundColor(1,1,1,v1); //设着两个窗口的背景色
    viewer->setBackgroundColor(1,1,1,v2);
    viewer->addCoordinateSystem(0.5, "reference", v1);
    viewer->addCoordinateSystem(0.5, "reference", v2);
}

void add_pcd(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer)
{
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb_object_all(cloud_all.makeShared());
  viewer->removeAllPointClouds();
  viewer->addPointCloud<pcl::PointXYZRGBA>(cloud_all.makeShared(), rgb_object_all, "poind_cloud_all", v1);
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "poind_cloud_all", v1);
  viewer->spinOnce(10);
  for(int i=0;i<CLASS_UPPER_LIMIT;i++)
  {
    ss.clear();
    ss.str("");
    ss<<"object_"<<i;
    ss>>string_name;
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb_object(object_visiable[i].makeShared());
    viewer->addPointCloud<pcl::PointXYZRGBA>(object_visiable[i].makeShared(), rgb_object, string_name, v2);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, string_name, v2);
  }
}

void update_pcd(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer)
{
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb_object_all(cloud_all.makeShared());
  viewer->updatePointCloud<pcl::PointXYZRGBA>(cloud_all.makeShared(), rgb_object_all, "poind_cloud_all");
  for(int i=0;i<CLASS_UPPER_LIMIT;i++)
  {
    ss.clear();
    ss.str("");
    ss<<"object_"<<i;
    ss>>string_name;
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb_object(object_visiable[i].makeShared());
    viewer->updatePointCloud<pcl::PointXYZRGBA>(object_visiable[i].makeShared(), rgb_object, string_name);
  }
}


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


void callback(const sensor_msgs::PointCloud2ConstPtr& input_cloud, boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer, ros::NodeHandle nh)
{
  my_num++;
  if (my_num%60==1)
  {
    ss.clear();
    ss.str("");
    ss << "/tmp/infer/" << my_num;
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
      listener.lookupTransform("robot1/camera_rgb_optical_frame", "robot1/odom", ros::Time(0), transform);
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
    matrix_now = t.inverse().matrix();

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>());
    pcl::fromROSMsg(*input_cloud, *cloud);
    pcl::toROSMsg(*input_cloud, image);
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8);
    imwrite(file_name + "/image.png", cv_ptr->image);
    ros::ServiceClient client = nh.serviceClient<semantic_mapping::MaskInfer>("infer_image");
    semantic_mapping::MaskInfer srv;
    srv.request.num = my_num;

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
      pcl::transformPointCloud( *cloud, *cloud, matrix_now);
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
                    isfinite(cloud->points[i_row*cv_image.cols + i_col].x)  &&
                    isfinite(cloud->points[i_row*cv_image.cols + i_col].y)  &&
                    isfinite(cloud->points[i_row*cv_image.cols + i_col].z)
                    )
                  indexs.push_back(i_row*cv_image.cols + i_col);
            }
        }
        pcl::copyPointCloud(*cloud, indexs, objects_temp);
        if (objects[class_data[i]].points.size()==0)
          objects[class_data[i]] = objects_temp;
        else
          objects[class_data[i]] += objects_temp;
      }
      for(int i_model=0;i_model<class_num;i_model++)
      {
        int class_index = class_data[i_model];
        object_visiable[class_index].clear();
        object_visiable[class_index].width    = objects[class_index].width;
        object_visiable[class_index].height   = objects[class_index].height;
        object_visiable[class_index].is_dense = true;
        object_visiable[class_index].points.resize (object_visiable[class_index].width * object_visiable[class_index].height);
        cout << "Class "<< class_index << ": "<< object_visiable[class_index].points.size() <<" points"<<endl;
        for(int i_points=0;i_points<object_visiable[class_index].points.size();i_points++)
        {
          object_visiable[class_index].points[i_points].x = objects[class_index].points[i_points].x;
          object_visiable[class_index].points[i_points].y = objects[class_index].points[i_points].y;
          object_visiable[class_index].points[i_points].z = objects[class_index].points[i_points].z;
          object_visiable[class_index].points[i_points].b = bgr_list[class_index][0];
          object_visiable[class_index].points[i_points].g = bgr_list[class_index][1];
          object_visiable[class_index].points[i_points].r = bgr_list[class_index][2];
        }
        pcl::PCLPointCloud2::Ptr output_part (new pcl::PCLPointCloud2 ());
        pcl::toPCLPointCloud2(object_visiable[class_index],*output_part);
        pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
        sor.setInputCloud(output_part);
        sor.setLeafSize(0.05, 0.05, 0.05);
        sor.filter(*output_part);
        pcl::fromPCLPointCloud2(*output_part, object_visiable[class_index]);
        ss.clear();
        ss.str("");
        ss<<file_name << "/" << class_index<<".pcd";
        ss>>file_name_pcd;
        pcl::io::savePCDFile(file_name_pcd, object_visiable[class_index]);
      }
//      pcl::PCLPointCloud2::Ptr output (new pcl::PCLPointCloud2 ());
//      pcl::toPCLPointCloud2(*cloud, *output);
//      pcl::VoxelGrid<pcl::PCLPointCloud2> sor_output;
//      sor_output.setInputCloud(output);
//      sor_output.setLeafSize(0.05, 0.05, 0.05);
//      sor_output.filter(*output);
//      pcl::fromPCLPointCloud2(*output, *cloud);

      if (my_num==1)
      {}
      else if(isFirst)
      {
        cloud_all = *cloud;
        add_pcd(viewer);
        isFirst = false;
      }
      else
      {
        cloud_all += *cloud;
        update_pcd(viewer);
      }
      ss.clear();
      ss.str("");
      ss<<file_name << "/all.pcd";
      ss>>file_name_pcd;
      pcl::io::savePCDFile(file_name_pcd, cloud_all);
      viewer->spinOnce(1000);
      boost::this_thread::sleep(boost::posix_time::microseconds(1000));
    }
    else
    {
      ROS_ERROR("Failed to call service add_two_ints");
    }
//    viewer->saveCameraParameters("camera.cam");
    //save pointcloud

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
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer ("3D Viewer"));
  char* cam_file = argv[1];
  simpleVis (viewer, cam_file);
  ros::Subscriber realsense = nh.subscribe<sensor_msgs::PointCloud2>("/robot1/camera/depth/points", 1, boost::bind(&callback, _1, viewer, nh));
  ros::spin();
  return 0;
}
