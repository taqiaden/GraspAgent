#!/bin/bash

source /home/yumi/yumi_egm_ws/devel/setup.bash
rosservice call /phoxi_camera/get_frame "in_: -1"

