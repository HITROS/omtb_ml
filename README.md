# OMTB_ML
Machine Learning for OpenManipulator with TurtleBot3(OMTB)

## Environment

System: Ubuntu 16.04

ROS version: Kinetic

Gazebo version: 7.0 or higher

### AutoLabel

This is a tool for collecting coco dataset in Gazebo environment.

1. Launch camera

```
roslaunch camera_gazebo empty_world.launch d1:=$(D1) d2:=$(D2) d3:=$(D3) d4:=$(D4) d5:=$(D5)
```

2. Create object & take photo

```
roslaunch auto_label save_pic.launch dir:=$(Your path) file:=$(Your SDF file) model:=$(Model name)
```

3. Remove object

```
roslaunch auto_label delete.launch
```

4. Create coco dataset

```
roslaunch auto_label create_coco.launch dir:=$(Your path)
```

### Semantic Mapping

3D semantic mapping in Gazebo environment

![image](https://github.com/HITROS/omtb_ml/blob/master/semantic_mapping/map.png)

1. Launch robot

```
roslaunch omtb_gazebo 1tb_room2.launch
```

2. Start service
```
roslaunch semantic_mapping infer.launch
```

3. Launch controller

```
roslaunch omtb_control turtlebot3_key.launch
```

4. Start client

```
roslaunch semantic_mapping mapping.launch
```
