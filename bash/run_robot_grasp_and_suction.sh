#!/bin/bash

source /home/yumi/yumi_egm_ws/devel/setup.bash
rostopic pub -1 /execute_robot std_msgs/Int32 "data: 2"