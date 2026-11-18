#!/usr/bin/env bash
# Downloads the three public evaluation videos and the KITTI detection dataset (YOLO format).
set -e
mkdir -p data datasets
curl -L -o data/bay_traffic.mp4 https://raw.githubusercontent.com/Bayunova28/Vehicle_Counting_Detection/master/Traffic.mp4
curl -L -o data/bridge.mp4 https://raw.githubusercontent.com/MicrocontrollersAndMore/OpenCV_3_Car_Counting_Cpp/master/CarsDrivingUnderBridge.mp4
curl -L -o data/az_input.mp4 https://raw.githubusercontent.com/ahmetozlu/vehicle_counting_tensorflow/master/input_video.mp4
curl -L -o datasets/kitti.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/kitti.zip
unzip -q datasets/kitti.zip -d datasets/kitti
echo "Done. Next: python make_kitti_subset.py datasets/kitti datasets/kitti_sub"
