#!/usr/bin/bash

if [ "$1" == "build" ]; then
python setup.py build_ext develop
RUN_TYPE="$2"
else
RUN_TYPE="$1"
fi

./tools/dist_deploy.sh ./configs/softgroup_deploy.yaml softgroup_scannet_spconv2.pth 1 $RUN_TYPE
