"""Generate the repository README from the saved result files, so every number matches the outputs."""
import json
from configs import VIDEOS

FM = json.load(open("results/forecast_metrics.json")); SW = json.load(open("results/freeway_line_sweep.json"))
TS = json.load(open("results/train_summary.json"))
c = TS["config"]; m = TS["metrics"]

def row(*xs): return "| " + " | ".join(str(x) for x in xs) + " |"
pooled_n = sum(FM[k]["ttl_mae_all_s"]["n"] for k in VIDEOS)
pool = {k: sum(FM[v]["ttl_mae_all_s"][k] * FM[v]["ttl_mae_all_s"]["n"] for v in VIDEOS) / pooled_n for k in ("linear", "const_accel", "ours_inverse_depth")}

md = f"""# Traffic flow nowcasting with Ultralytics YOLO26

Single-camera pipeline that **detects, tracks and counts vehicles** crossing a line, and **forecasts the next few seconds of traffic flow** with calibrated uncertainty.

- **Detection:** Ultralytics YOLO26s (COCO weights) for the demos; a YOLO26n fine-tuned on KITTI is included as a training experiment
- **Tracking:** ByteTrack
- **Counting:** each vehicle is counted once per line crossing, in both directions, with parked vehicles filtered out
- **Forecasting (the new part):** inverse-depth time-to-line forecasts, online adaptive conformal intervals and a probabilistic next-3 s count forecast

![Time-to-line error](figs/fig_ttl_mae.png)

## What's new

1. **Calibration-free time-to-line (TTL) forecasts.** On a flat road seen by a pinhole camera, the image row *y* of a vehicle's ground-contact point satisfies *(y − y_h) ∝ 1/D*. At constant speed, *u(t) = 1/(y(t) − y_h)* is therefore linear in time. The horizon row *y_h* is fitted from the tracks themselves (first 40% of each video), and *u* is extrapolated to the counting line: *TTL = (u_line − u_now)/a*.
2. **Calibrated per-vehicle uncertainty.** Online adaptive conformal inference (Gibbs & Candès, 2021) is applied to normalised TTL residuals. Only vehicles that have already crossed are used, so there is no look-ahead.
3. **Probabilistic flow nowcast.** Each approaching vehicle's conformal predictive distribution gives P(cross within 3 s). These are combined (Poisson-binomial), plus a Poisson term for vehicles not yet in view, giving the expected count with an 80% interval.

> Conformal prediction is already used for network-level traffic forecasting from sensors. The angle here is per-vehicle, single-camera forecasts aggregated into a calibrated line-count forecast. A quick search found no direct match, but this is not a claim that it has never been done.

## Datasets

| Purpose | Dataset | Details |
|---|---|---|
| Fine-tuning | **KITTI** (Ultralytics YOLO release, `kitti.zip`) | full: 5,985 train / 1,496 val, 8 classes; used: **800 train / 200 val** (random, seed 0) |
| Evaluation | {VIDEOS['urban']['title']} | `{VIDEOS['urban']['source']}` (1280×720, 25 fps, 81 s) |
| Evaluation | {VIDEOS['freeway']['title']} | `{VIDEOS['freeway']['source']}` (1280×720, 30 fps, 33 s) |
| Evaluation | {VIDEOS['residential']['title']} | `{VIDEOS['residential']['source']}` (640×352, 24 fps, first 30 s) |

The videos belong to their original repositories and are downloaded by `scripts/download_data.sh`, not redistributed here.

## Training (KITTI fine-tuning)

{row("Setting", "Value")}
{row("---", "---")}
{row("Model", "YOLO26n, initialised from COCO weights")}
{row("Epochs / image size / batch", f"{c['epochs']} / {c['imgsz']} px / {c['batch']}")}
{row("Optimizer", c['optimizer'] + " (Ultralytics auto selection)")}
{row("Hardware", "CPU only, 2 cores")}
{row("Training time", f"{TS['train_minutes']:.0f} min")}

**Validation (200 KITTI images):** precision {m['precision']:.3f} · recall {m['recall']:.3f} · mAP50 **{m['mAP50']:.3f}** · mAP50-95 **{m['mAP50-95']:.3f}**

{row("Class", "AP50", "AP50-95")}
{row("---", "---", "---")}
""" + "\n".join(row(k, f"{v['AP50']:.3f}", f"{v['AP50-95']:.3f}") for k, v in sorted(TS["per_class"].items(), key=lambda kv: -kv[1]["AP50"])) + f"""

![Training curves](figs/fig_train_curves.png)

This is a small CPU run (800 images, {c['epochs']} epochs) meant to show the workflow. A GPU and the full dataset would give much higher mAP.

## Forecasting results (held-out last 60% of each video)

**Time-to-line mean absolute error (s)**

{row("Video", "forecasts", "Linear", "Const. accel.", "**Ours**")}
{row("---", "---", "---", "---", "---")}
""" + "\n".join(row(VIDEOS[k]["title"], FM[k]["ttl_mae_all_s"]["n"], FM[k]["ttl_mae_all_s"]["linear"], FM[k]["ttl_mae_all_s"]["const_accel"], f"**{FM[k]['ttl_mae_all_s']['ours_inverse_depth']}**") for k in VIDEOS) + f"""
{row("**Pooled**", pooled_n, f"{pool['linear']:.3f}", f"{pool['const_accel']:.3f}", f"**{pool['ours_inverse_depth']:.3f}**")}

**Conformal intervals and next-3 s count forecast**

{row("Video", "line crossings", "80% interval coverage", "mean width (s)", "count MAE: persistence", "count MAE: **ours**")}
{row("---", "---", "---", "---", "---", "---")}
""" + "\n".join(row(VIDEOS[k]["title"], FM[k]["crossings"], f"{FM[k]['conformal']['coverage']:.0%}", FM[k]["conformal"]["mean_width_s"], FM[k]["nowcast"]["mae_persistence"], f"**{FM[k]['nowcast']['mae_ours']}**") for k in VIDEOS) + """

**Freeway line-position sensitivity.** Forecasting only helps when vehicles are visible long enough before the line:

""" + row("Line row", "median visible time before crossing (s)", "MAE persistence", "MAE ours") + "\n" + row("---", "---", "---", "---") + "\n" + \
"\n".join(row(r["line_y"], r["lead_median_s"], r["mae_persistence"], r["mae_ours"]) for r in SW) + """

![Conformal and nowcast](figs/fig_conformal_nowcast.png)
![Nowcast timeline](figs/fig_nowcast_timeline.png)
![Line sweep](figs/fig_line_sweep.png)
![Heatmap](figs/fig_heatmap.jpg)

## Run it

```bash
pip install -r requirements.txt
bash scripts/download_data.sh          # 3 public videos + KITTI (390 MB)
bash scripts/run_all.sh                # tracking -> analysis -> training -> demo video
```

| File | Purpose |
|---|---|
| `track.py` | YOLO26 + ByteTrack, saves every tracked box per frame |
| `analysis.py` | counting, parked filter, horizon fit, TTL forecasts, adaptive conformal, nowcast, metrics |
| `configs.py` | video sources and counting lines |
| `figs_analysis.py` | runs the analysis on all videos, writes `results/*.json` and the figures |
| `make_kitti_subset.py`, `train.py` | KITTI subset and YOLO26n fine-tuning + validation |
| `make_video.py`, `build_video.py` | renders the demo video |

## Limitations

- Three short clips (about 2.4 minutes analysed). The results are encouraging but not conclusive.
- Per-vehicle interval coverage was 74–80% against the 80% target; it slightly under-covers on busy scenes.
- The flat-road, roughly-constant-speed assumption will need extending for hills and stop-and-go traffic.
- The counting line is placed by hand for each video, and forecast skill depends on how long vehicles are visible before it.

## Acknowledgements

Ultralytics YOLO26 and ByteTrack; the KITTI Vision Benchmark Suite; the authors of the three public sample videos listed above.
"""
open("README_repo.md", "w").write(md); print("README written", len(md))
