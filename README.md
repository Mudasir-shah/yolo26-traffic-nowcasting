<div align="center">

# 🚦 Traffic Flow Nowcasting with Ultralytics YOLO26

**Detect · Track · Count · Forecast: short-term traffic flow from a single camera, with calibrated uncertainty**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO26-111F68)](https://docs.ultralytics.com/)
[![Tracker](https://img.shields.io/badge/Tracker-ByteTrack-22D3EE)](https://github.com/ifzhang/ByteTrack)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<img src="assets/demo_preview.gif" width="900" alt="YOLO26 traffic counting and flow forecasting demo">

[🎥 Full demo video](https://github.com/Mudasir-shah/yolo26-traffic-nowcasting/releases) · [📊 Results](#-forecasting-results-held-out-last-60-of-each-video) · [🏋️ Training](#️-training-kitti-fine-tuning) · [🚀 Run it](#-run-it)

</div>

## ✨ Highlights

- **Time-to-line error of 0.17 s**: 61% lower than a constant-acceleration baseline (3,497 held-out forecasts, 3 public videos)
- **Calibrated uncertainty**: every vehicle gets an 80% conformal interval for when it will cross the line
- **Next-3 s flow forecast**: beats the persistence baseline on all three videos
- **No camera calibration**: road perspective is learned from the vehicle tracks themselves
- **Trained and evaluated end to end**: YOLO26 fine-tuned on KITTI, with full training curves, PR curves and a confusion matrix

## 📌 Overview


Single-camera pipeline that **detects, tracks and counts vehicles** crossing a line, and **forecasts the next few seconds of traffic flow** with calibrated uncertainty.

- **Detection:** Ultralytics YOLO26s (COCO weights) for the demos; a YOLO26n fine-tuned on KITTI is included as a training experiment
- **Tracking:** ByteTrack
- **Counting:** each vehicle is counted once per line crossing, in both directions, with parked vehicles filtered out
- **Forecasting (the new part):** inverse-depth time-to-line forecasts, online adaptive conformal intervals and a probabilistic next-3 s count forecast

![Time-to-line error](figs/fig_ttl_mae.png)

## 💡 What's new

1. **Calibration-free time-to-line (TTL) forecasts.** On a flat road seen by a pinhole camera, the image row *y* of a vehicle's ground-contact point satisfies *(y − y_h) ∝ 1/D*. At constant speed, *u(t) = 1/(y(t) − y_h)* is therefore linear in time. The horizon row *y_h* is fitted from the tracks themselves (first 40% of each video), and *u* is extrapolated to the counting line: *TTL = (u_line − u_now)/a*.
2. **Calibrated per-vehicle uncertainty.** Online adaptive conformal inference (Gibbs & Candès, 2021) is applied to normalised TTL residuals. Only vehicles that have already crossed are used, so there is no look-ahead.
3. **Probabilistic flow nowcast.** Each approaching vehicle's conformal predictive distribution gives P(cross within 3 s). These are combined (Poisson-binomial), plus a Poisson term for vehicles not yet in view, giving the expected count with an 80% interval.

> Conformal prediction is already used for network-level traffic forecasting from sensors. The angle here is per-vehicle, single-camera forecasts aggregated into a calibrated line-count forecast. A quick search found no direct match, but this is not a claim that it has never been done.

## 🗂️ Datasets

| Purpose | Dataset | Details |
|---|---|---|
| Fine-tuning | **KITTI** (Ultralytics YOLO release, `kitti.zip`) | full: 5,985 train / 1,496 val, 8 classes; used: **800 train / 200 val** (random, seed 0) |
| Evaluation | Urban multi-lane road | `github.com/Bayunova28/Vehicle_Counting_Detection` (1280×720, 25 fps, 81 s) |
| Evaluation | Freeway seen from a bridge | `github.com/MicrocontrollersAndMore/OpenCV_3_Car_Counting_Cpp` (1280×720, 30 fps, 33 s) |
| Evaluation | Residential street | `github.com/ahmetozlu/vehicle_counting_tensorflow` (640×352, 24 fps, first 30 s) |

The videos belong to their original repositories and are downloaded by `scripts/download_data.sh`, not redistributed here.

## 🏋️ Training (KITTI fine-tuning)

| Setting | Value |
| --- | --- |
| Model | YOLO26n, initialised from COCO weights |
| Epochs / image size / batch | 15 / 416 px / 16 |
| Optimizer | auto (Ultralytics auto selection) |
| Hardware | CPU only, 2 cores |
| Training time | 42 min |

**Validation (200 KITTI images):** precision 0.548 · recall 0.318 · mAP50 **0.358** · mAP50-95 **0.211**

| Class | AP50 | AP50-95 |
| --- | --- | --- |
| car | 0.796 | 0.491 |
| truck | 0.636 | 0.432 |
| pedestrian | 0.502 | 0.238 |
| van | 0.334 | 0.240 |
| tram | 0.276 | 0.181 |
| cyclist | 0.155 | 0.053 |
| misc | 0.095 | 0.047 |
| person_sitting | 0.075 | 0.010 |

![Training curves](figs/fig_train_curves.png)

<p align="center"><img src="figs/training/BoxPR_curve.png" width="48%"> <img src="figs/training/confusion_matrix_normalized.png" width="48%"></p>
<p align="center"><img src="figs/training/val_batch0_pred.jpg" width="90%"><br><em>YOLO26n predictions on KITTI validation images</em></p>

This is a small CPU run (800 images, 15 epochs) meant to show the workflow. A GPU and the full dataset would give much higher mAP.

## 📊 Forecasting results (held-out last 60% of each video)

**Time-to-line mean absolute error (s)**

| Video | forecasts | Linear | Const. accel. | **Ours** |
| --- | --- | --- | --- | --- |
| Urban multi-lane road | 2459 | 0.523 | 0.498 | **0.215** |
| Freeway seen from a bridge | 886 | 0.772 | 0.216 | **0.03** |
| Residential street | 152 | 1.535 | 0.501 | **0.162** |
| **Pooled** | 3497 | 0.630 | 0.427 | **0.166** |

**Conformal intervals and next-3 s count forecast**

| Video | line crossings | 80% interval coverage | mean width (s) | count MAE: persistence | count MAE: **ours** |
| --- | --- | --- | --- | --- | --- |
| Urban multi-lane road | 83 | 74% | 2.673 | 1.313 | **0.915** |
| Freeway seen from a bridge | 86 | 80% | 0.075 | 2.673 | **1.991** |
| Residential street | 8 | 75% | 0.848 | 0.567 | **0.219** |

**Freeway line-position sensitivity.** Forecasting only helps when vehicles are visible long enough before the line:

| Line row | median visible time before crossing (s) | MAE persistence | MAE ours |
| --- | --- | --- | --- |
| 110 | 1.63 | 3.068 | 2.419 |
| 150 | 1.33 | 2.673 | 1.991 |
| 190 | 1.03 | 2.048 | 2.32 |
| 250 | 0.73 | 2.03 | 2.437 |

![Conformal and nowcast](figs/fig_conformal_nowcast.png)
![Nowcast timeline](figs/fig_nowcast_timeline.png)
![Line sweep](figs/fig_line_sweep.png)
![Heatmap](figs/fig_heatmap.jpg)

## 🚀 Run it

```bash
pip install -r requirements.txt
bash scripts/download_data.sh          # 3 public videos + KITTI (390 MB)
bash scripts/run_all.sh                # tracking -> analysis -> training -> demo video
# GPU (e.g. RTX 4090): open traffic_nowcasting_gpu.ipynb and run the cells top to bottom
```

| File | Purpose |
|---|---|
| `track.py` | YOLO26 + ByteTrack, saves every tracked box per frame |
| `analysis.py` | counting, parked filter, horizon fit, TTL forecasts, adaptive conformal, nowcast, metrics |
| `configs.py` | video sources and counting lines |
| `figs_analysis.py` | runs the analysis on all videos, writes `results/*.json` and the figures |
| `make_kitti_subset.py`, `train.py` | KITTI subset and YOLO26n fine-tuning + validation |
| `make_video.py`, `build_video.py` | renders the demo video |

## ⚠️ Limitations

- Three short clips (about 2.4 minutes analysed). The results are encouraging but not conclusive.
- Per-vehicle interval coverage was 74–80% against the 80% target; it slightly under-covers on busy scenes.
- The flat-road, roughly-constant-speed assumption will need extending for hills and stop-and-go traffic.
- The counting line is placed by hand for each video, and forecast skill depends on how long vehicles are visible before it.

## 🙏 Acknowledgements

Ultralytics YOLO26 and ByteTrack; the KITTI Vision Benchmark Suite; the authors of the three public sample videos listed above.

## 📜 Citation

```bibtex
@software{shah2026trafficnowcasting,
  author = {Shah, Sayyed Ajmal},
  title  = {Traffic Flow Nowcasting with Ultralytics YOLO26},
  year   = {2026},
  url    = {https://github.com/Mudasir-shah/yolo26-traffic-nowcasting}
}
```


## 👤 Author

**Sayyed Ajmal Shah**: computer vision research on road infrastructure inspection and traffic analysis · [GitHub](https://github.com/Mudasir-shah)
