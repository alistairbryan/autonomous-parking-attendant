#!/bin/bash

source ~/home/chloe/353_ws/devel/setup.bash
export ROS_MASTER_URI=http://localhost:11311

export ROS_IP=127.0.0.1  

python /home/chloe/353_ws/src/2019F_competition_students/enph353/enph353_gazebo/nodes/drive.py