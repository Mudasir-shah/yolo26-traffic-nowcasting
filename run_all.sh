#!/usr/bin/env bash
# Full pipeline: tracking -> forecasting analysis + figures -> KITTI fine-tuning -> demo video.
set -e
python track.py data/bay_traffic.mp4 tracks_bay.json 81
python track.py data/bridge.mp4 tracks_bridge.json 33
python track.py data/az_input.mp4 tracks.json 30
python figs_analysis.py
python make_kitti_subset.py datasets/kitti datasets/kitti_sub
mkdir -p train && (cd train && python ../train.py ../kitti_sub.yaml 15 416)
python build_video.py render $(seq 0 21)
python build_video.py concat traffic_nowcasting_demo.mp4
