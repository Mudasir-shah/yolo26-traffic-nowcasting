"""Storyboard for the research demo video. Usage:
    python build_video.py render <scene indices...>   # render selected scenes to parts/
    python build_video.py still <scene> <t>           # save a still for checking
    python build_video.py concat out.mp4              # join all parts
"""
import json, os, sys, pickle, csv
from make_video import *
from configs import VIDEOS

os.makedirs(PARTS, exist_ok=True)
TS = json.load(open("train/train_summary.json")) if os.path.exists("train/train_summary.json") else None
FM = json.load(open("results/forecast_metrics.json"))
SWEEP = json.load(open("results/freeway_line_sweep.json"))

def A(key):
    return pickle.load(open(f"results/cache_{key}_{VIDEOS[key]['line'][0]}.pkl", "rb"))

def heading(kicker, title, y=120, size=60):
    return [(spaced(kicker, F("semi", 24), VIOLET), 96, y, 0.0), (tblock(title, F("bold", size), WHITE, 1700), 96, y + 44, 0.1)]

# ---------------------------------------------------------------- slides
def s_title():
    els = [(spaced("COMPUTER VISION PROJECT", F("semi", 28), VIOLET), 120, 250, 0)]
    els.append((tblock("Traffic flow nowcasting\nwith YOLO26", F("bold", 96), WHITE, 1700, lh=118), 120, 300, .12))
    els.append((tblock("From detection and tracking to calibrated short-term forecasts of how many vehicles will cross a line",
                       F("reg", 34), MUTED, 1300), 120, 560, .25))
    x = 120
    for i, (t, c) in enumerate([("Ultralytics YOLO26", CYAN), ("ByteTrack", CYAN), ("Adaptive conformal prediction", VIOLET), ("3 public traffic videos", LIME)]):
        ch = chip(t, c); els.append((ch, x, 700, .4 + .08 * i)); x += ch.width + 18
    return els

def s_new():
    els = heading("CONTRIBUTIONS", "What this project adds")
    cards = [("01", "Calibration-free time-to-line forecasts",
              "On a flat road, 1 / (y − y_horizon) grows linearly in time at constant speed. The horizon row is learned from the tracks themselves, so no camera calibration is needed."),
             ("02", "Calibrated uncertainty for every vehicle",
              "Online adaptive conformal intervals (80%) built only from vehicles that have already crossed, so there is no look-ahead."),
             ("03", "Probabilistic flow nowcast",
              "Per-vehicle crossing probabilities are summed into the expected count for the next 3 s, plus an 80% interval and a term for vehicles not yet in view.")]
    for i, (n, t, b) in enumerate(cards):
        c = card(552, 470); d = ImageDraw.Draw(c)
        d.text((36, 28), n, font=F("bold", 64), fill=CYAN)
        for j, l in enumerate(wrap(t, F("bold", 32), 480)): d.text((36, 128 + j * 42), l, font=F("bold", 32), fill=WHITE)
        for j, l in enumerate(wrap(b, F("reg", 25), 480)): d.text((36, 236 + j * 35), l, font=F("reg", 25), fill=MUTED)
        els.append((c, 96 + i * 584, 300, .3 + .15 * i))
    els.append((tblock("Note: conformal prediction is already used for network-level traffic forecasting from sensors. The angle here is single-camera, per-vehicle forecasts rolled up into a calibrated count forecast. I found no direct match in a quick search, but this is not a claim that it has never been done.",
                       F("reg", 22), GREY, 1700), 96, 820, .9))
    return els

def s_pipeline():
    els = heading("METHOD", "Pipeline")
    steps = [("Video frames", "any fixed camera", MUTED), ("YOLO26", "vehicle detection", CYAN), ("ByteTrack", "persistent IDs", CYAN),
             ("Line counting", "both directions,\nparked filter", LIME), ("Inverse-depth TTL", "time-to-line\nper vehicle", VIOLET), ("Conformal nowcast", "80% intervals,\nnext-3 s count", VIOLET)]
    bw, gap, y = 248, 46, 330
    for i, (t, sub, col) in enumerate(steps):
        c = card(bw, 200, edge=col); d = ImageDraw.Draw(c)
        for j, l in enumerate(wrap(t, F("bold", 28), bw - 30)): d.text((20, 26 + j * 36), l, font=F("bold", 28), fill=WHITE)
        for j, l in enumerate(sub.split("\n")): d.text((20, 112 + j * 30), l, font=F("reg", 22), fill=MUTED)
        x = 96 + i * (bw + gap); els.append((c, x, y, .25 + .18 * i))
        if i < len(steps) - 1:
            ar = Image.new("RGBA", (gap, 40), (0, 0, 0, 0)); ImageDraw.Draw(ar).polygon([(8, 8), (gap - 8, 20), (8, 32)], fill=EDGE + (255,))
            els.append((ar, x + bw, y + 80, .3 + .18 * i))
    f = card(1728, 250); d = ImageDraw.Draw(f)
    d.text((40, 30), "Forecast model", font=F("semi", 24), fill=VIOLET)
    d.text((40, 74), "u(t) = 1 / (y(t) − y_h)  ≈  a·t + b        TTL = (u_line − u_now) / a", font=F("bold", 36), fill=WHITE)
    d.text((40, 140), "y(t): bottom edge of the vehicle box (image row)   ·   y_h: horizon row, fitted on the first 40% of each video",
           font=F("reg", 24), fill=MUTED)
    d.text((40, 180), "All forecast metrics are reported on the remaining 60%. Baselines: linear and constant-acceleration extrapolation of y(t).",
           font=F("reg", 24), fill=MUTED)
    els.append((f, 96, 600, 1.4))
    return els

def s_data():
    els = heading("DATA", "Datasets used")
    L = card(820, 640); d = ImageDraw.Draw(L)
    d.text((36, 30), "TRAINING · KITTI (Ultralytics release)", font=F("semi", 24), fill=VIOLET)
    lines = [("Full dataset", "5,985 train / 1,496 val images, 8 classes"), ("Used here", (TS or {}).get("data_desc", "800 train / 200 val (random sample, seed 0)")),
             ("Image size", "1242 × 375 road scenes from a car-mounted camera"), ("Hardware", (TS or {}).get("hardware", "CPU (2 cores)"))]
    y = 84
    for k, v in lines:
        d.text((36, y), k, font=F("semi", 24), fill=WHITE); d.text((250, y), v, font=F("reg", 24), fill=MUTED); y += 48
    d.text((36, y + 14), "Boxes in the 800-image training subset", font=F("semi", 22), fill=VIOLET); y += 56
    counts = [("car", 3102), ("pedestrian", 499), ("van", 295), ("cyclist", 168), ("truck", 125), ("misc", 91), ("tram", 69), ("person sitting", 19)]
    for k, v in counts:
        d.text((36, y), k, font=F("reg", 21), fill=MUTED)
        d.rounded_rectangle([230, y + 4, 230 + int(480 * v / 3102), y + 22], 5, fill=CYAN); d.text((236 + int(480 * v / 3102) + 6, y), f"{v:,}", font=F("semi", 21), fill=WHITE)
        y += 31
    els.append((L, 96, 300, .3))
    R = card(880, 640); d = ImageDraw.Draw(R)
    d.text((36, 30), "EVALUATION · 3 public traffic videos (GitHub)", font=F("semi", 24), fill=VIOLET)
    y = 86
    for key in ("urban", "freeway", "residential"):
        v = VIDEOS[key]; cap = cv2.VideoCapture(v["path"]); w_, h_, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5); n = cap.get(7)
        d.text((36, y), v["title"], font=F("bold", 28), fill=WHITE)
        d.text((36, y + 42), f"{w_}×{h_}  ·  {fps:.0f} fps  ·  {n / fps:.0f} s  ·  {FM[key]['crossings']} line crossings", font=F("reg", 23), fill=MUTED)
        d.text((36, y + 76), v["source"], font=F("med", 21), fill=CYAN); y += 178
    els.append((R, 944, 300, .5))
    return els

def s_train_setup():
    els = heading("TRAINING", "Fine-tuning setup")
    import yaml
    args = yaml.safe_load(open(os.path.join(TS["save_dir"], "args.yaml")))
    rows = [("Model", f"{TS.get('model', 'yolo26n').upper().replace('YOLO', 'YOLO')} ({TS.get('params_m', 2.4)} M parameters), initialised from COCO weights"), ("Data", TS.get("data_desc", "KITTI subset · 800 train / 200 val · 8 classes")),
            ("Epochs", f"{args['epochs']}"), ("Image size", f"{args['imgsz']} px"), ("Batch size", f"{args['batch']}"),
            ("Optimizer", f"{args['optimizer']} (Ultralytics auto selection), lr0 {args['lr0']}, momentum {args['momentum']}"),
            ("Augmentation", f"mosaic {args['mosaic']}, horizontal flip {args['fliplr']}, HSV jitter, scale {args['scale']}"),
            ("Hardware", TS.get("hardware", "CPU (2 cores)")), ("Training time", f"{TS['train_minutes']:.0f} minutes")]
    c = card(1728, 620); d = ImageDraw.Draw(c); y = 34
    for k, v in rows:
        d.text((44, y), k, font=F("semi", 28), fill=VIOLET); d.text((380, y), v, font=F("reg", 28), fill=WHITE); y += 62
    els.append((c, 96, 300, .3))
    return els

def s_fig(kicker, title, path, note=None, white=False, maxh=640):
    def b():
        els = heading(kicker, title)
        im = fit_image(path, 1500 if note is None else 1180, maxh, white_card=white)
        x = 96 if note else (W - im.width) // 2
        els.append((im, x, 290, .25, "wipe"))
        if note: els.append((tblock(note, F("reg", 28), MUTED, 460), 1330, 320, 1.2))
        return els
    return b

def s_val():
    els = heading("TRAINING", f"Validation results ({TS.get('val_images', 200)} KITTI val images)")
    m = TS["metrics"]; x = 96
    for k in ("precision", "recall", "mAP50", "mAP50-95"):
        c = card(250, 150); d = ImageDraw.Draw(c); d.text((26, 18), k, font=F("semi", 24), fill=VIOLET); d.text((26, 56), f"{m[k]:.3f}", font=F("bold", 52), fill=WHITE)
        els.append((c, x, 290, .2)); x += 268
    t = card(1054, 460); d = ImageDraw.Draw(t)
    d.text((30, 22), "Class", font=F("semi", 24), fill=VIOLET); d.text((420, 22), "AP50", font=F("semi", 24), fill=VIOLET); d.text((620, 22), "AP50-95", font=F("semi", 24), fill=VIOLET)
    y = 66
    for cn, v in sorted(TS["per_class"].items(), key=lambda kv: -kv[1]["AP50"]):
        d.text((30, y), cn, font=F("reg", 24), fill=WHITE); d.text((420, y), f"{v['AP50']:.3f}", font=F("semi", 24), fill=WHITE)
        d.text((620, y), f"{v['AP50-95']:.3f}", font=F("semi", 24), fill=WHITE)
        d.rounded_rectangle([800, y + 6, 800 + int(220 * v["AP50"]), y + 22], 4, fill=CYAN); y += 46
    els.append((t, 96, 470, .5))
    pr = os.path.join(TS["save_dir"], "BoxPR_curve.png")
    if not os.path.exists(pr): pr = os.path.join(TS["save_dir"], "PR_curve.png")
    els.append((fit_image(pr, 680, 620, white_card=True), 1180, 290, .8, "wipe"))
    return els

def s_conf():
    els = heading("TRAINING", "Confusion matrix and sample predictions")
    cm = os.path.join(TS["save_dir"], "confusion_matrix_normalized.png")
    els.append((fit_image(cm, 760, 650, white_card=True), 96, 280, .25, "wipe"))
    vb = os.path.join(TS["save_dir"], "val_batch0_pred.jpg")
    els.append((fit_image(vb, 900, 650, white_card=True), 900, 280, .6, "wipe"))
    return els

def s_democard(i, key):
    v = VIDEOS[key]; m = FM[key]
    def b():
        els = [(spaced(f"LIVE DEMO {i} / 3", F("semi", 30), VIOLET), 120, 360, 0), (tblock(v["title"], F("bold", 90), WHITE, 1700), 120, 410, .1)]
        els.append((tblock(f"Public video: {v['source']}", F("reg", 30), MUTED, 1600), 120, 540, .25))
        return els
    return b

def s_results_ttl():
    els = heading("RESULTS", "Time-to-line forecast error")
    els.append((fit_image("figs/fig_ttl_mae.png", 1180, 660), 96, 280, .25, "wipe"))
    allm = {k: FM[k]["ttl_mae_all_s"] for k in VIDEOS}
    n = sum(v["n"] for v in allm.values()); ours = sum(v["ours_inverse_depth"] * v["n"] for v in allm.values()) / n
    ca = sum(v["const_accel"] * v["n"] for v in allm.values()) / n; li = sum(v["linear"] * v["n"] for v in allm.values()) / n
    txt = (f"Pooled over {n:,} held-out forecasts:\n\nOurs  {ours:.2f} s\nConstant accel.  {ca:.2f} s\nLinear  {li:.2f} s\n\n"
           f"{(1 - ours / ca) * 100:.0f}% lower error than the stronger baseline, and the gap grows with lead time.")
    els.append((tblock(txt, F("reg", 30), WHITE, 470), 1330, 300, 1.2))
    return els

def s_results_now():
    els = heading("RESULTS", "Calibration and next-3 s flow forecast")
    els.append((fit_image("figs/fig_conformal_nowcast.png", 1728, 600), 96, 290, .25, "wipe"))
    return els

def s_sweep():
    els = heading("RESULTS", "When does forecasting help?")
    els.append((fit_image("figs/fig_line_sweep.png", 1180, 660), 96, 280, .25, "wipe"))
    els.append((tblock("On the freeway, cars appear from under the bridge just before the line. The forecast only beats persistence when vehicles are visible for more than about 1.3 s before crossing. Camera placement matters as much as the model.",
                       F("reg", 29), MUTED, 470), 1330, 300, 1.2))
    return els

def s_heat():
    els = heading("ANALYTICS", "Where the traffic flows (urban road)")
    els.append((fit_image("figs/fig_heatmap.jpg", 1180, 664), 96, 280, .25, "wipe"))
    els.append((tblock("Density of all moving-vehicle trajectories over the 81 s clip. The amber line is the counting line. Lane use and turning paths stand out without any manual labelling.",
                       F("reg", 29), MUTED, 470), 1330, 300, 1.0))
    return els

def s_limits():
    els = heading("HONEST LIMITS", "Limitations and next steps")
    items = [("Short clips", "3 videos, about 2.5 minutes in total. The results are encouraging, not conclusive. Longer multi-site footage is needed."),
             ("Calibration", "Per-vehicle interval coverage was 74–80% against an 80% target. Adaptive conformal is close, but it slightly under-covers on busy scenes."),
             ("Small CPU training run", "KITTI fine-tuning used 800 images on a CPU. A GPU and the full dataset would raise mAP considerably."),
             ("Flat-road assumption", "The inverse-depth model assumes a flat road and roughly constant speed. Hills and stop-and-go traffic will need extensions.")]
    for i, (t, b) in enumerate(items):
        c = card(852, 250); d = ImageDraw.Draw(c); d.text((32, 26), t, font=F("bold", 32), fill=WHITE)
        for j, l in enumerate(wrap(b, F("reg", 25), 780)): d.text((32, 86 + j * 36), l, font=F("reg", 25), fill=MUTED)
        els.append((c, 96 + (i % 2) * 876, 300 + (i // 2) * 276, .3 + .15 * i))
    return els

def s_outro():
    els = [(spaced("THANKS FOR WATCHING", F("semi", 30), VIOLET), 120, 330, 0),
           (tblock("Code, data sources and full results\nare in the project repository.", F("bold", 72), WHITE, 1700, lh=92), 120, 380, .12),
           (tblock("Which application would you use this for: signal timing, incident alerts, or planning?", F("reg", 34), MUTED, 1500), 120, 600, .3)]
    x = 120
    for i, t in enumerate(["#ComputerVision", "#YOLO26", "#Ultralytics", "#TrafficFlow", "#ConformalPrediction"]):
        ch = chip(t, CYAN if i % 2 == 0 else VIOLET); els.append((ch, x, 720, .45 + .06 * i)); x += ch.width + 16
    return els

def train_curves_fig():
    out = "figs/fig_train_curves.png"
    rows = list(csv.DictReader(open(os.path.join(TS["save_dir"], "results.csv"))))
    rows = [{k.strip(): float(v) for k, v in r.items()} for r in rows]
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import figs_style  # noqa: sets the dark style
    ep = [r["epoch"] for r in rows]
    fig, axs = plt.subplots(1, 3, figsize=(16, 6.2))
    for k, lab, col in (("train/box_loss", "train box", "#22d3ee"), ("val/box_loss", "val box", "#a78bfa")):
        if k in rows[0]: axs[0].plot(ep, [r[k] for r in rows], lw=2.6, color=col, label=lab, marker="o", ms=4)
    axs[0].set_title("Box loss"); axs[0].set_xlabel("epoch"); axs[0].legend()
    for k, lab, col in (("train/cls_loss", "train cls", "#22d3ee"), ("val/cls_loss", "val cls", "#a78bfa")):
        if k in rows[0]: axs[1].plot(ep, [r[k] for r in rows], lw=2.6, color=col, label=lab, marker="o", ms=4)
    axs[1].set_title("Classification loss"); axs[1].set_xlabel("epoch"); axs[1].legend()
    for k, lab, col in (("metrics/mAP50(B)", "mAP50", "#a3e635"), ("metrics/mAP50-95(B)", "mAP50-95", "#fbbf24"), ("metrics/precision(B)", "precision", "#64748b"), ("metrics/recall(B)", "recall", "#94a3b8")):
        if k in rows[0]: axs[2].plot(ep, [r[k] for r in rows], lw=2.6 if "mAP" in k else 1.6, color=col, label=lab, marker="o" if "mAP" in k else None, ms=4)
    axs[2].set_title("Validation metrics"); axs[2].set_xlabel("epoch"); axs[2].legend(loc="upper left", fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig); return out

def scenes():
    S = [Slide(6, "", s_title), Slide(12, "CONTRIBUTIONS", s_new), Slide(11, "METHOD", s_pipeline), Slide(11, "DATA", s_data)]
    if True:  # training slides always in the timeline (they need train/train_summary.json when rendered)
        S += [Slide(10, "TRAINING", s_train_setup),
              Slide(11, "TRAINING", s_fig("TRAINING", f"Training curves ({TS['config']['epochs'] if TS else 15} epochs, KITTI)", "figs/fig_train_curves.png")),
              Slide(11, "TRAINING", s_val), Slide(9, "TRAINING", s_conf)]
    S += [Slide(3.5, "LIVE DEMO", s_democard(1, "urban")),
          Demo(60, "LIVE DEMO 1/3", f"Live demo 1/3 · {VIDEOS['urban']['title']}", VIDEOS["urban"]["source"], VIDEOS["urban"]["path"], A("urban")),
          Slide(3.5, "LIVE DEMO", s_democard(2, "freeway")),
          Demo(32, "LIVE DEMO 2/3", f"Live demo 2/3 · {VIDEOS['freeway']['title']}", VIDEOS["freeway"]["source"], VIDEOS["freeway"]["path"], A("freeway")),
          Slide(3.5, "LIVE DEMO", s_democard(3, "residential")),
          Demo(30, "LIVE DEMO 3/3", f"Live demo 3/3 · {VIDEOS['residential']['title']}", VIDEOS["residential"]["source"], VIDEOS["residential"]["path"], A("residential")),
          Slide(12, "RESULTS", s_results_ttl),
          Slide(9, "RESULTS", s_fig("RESULTS", "Time-to-line error per video", "figs/fig_ttl_per_video.png")),
          Slide(11, "RESULTS", s_results_now),
          Slide(10, "RESULTS", s_fig("RESULTS", "Urban road: forecast vs reality", "figs/fig_nowcast_timeline.png", maxh=620)),
          Slide(11, "RESULTS", s_sweep), Slide(9, "ANALYTICS", s_heat), Slide(12, "LIMITATIONS", s_limits), Slide(7, "", s_outro)]
    return S

if __name__ == "__main__":
    cmd = sys.argv[1]
    if TS and not os.path.exists("figs/fig_train_curves.png"): train_curves_fig()
    S = scenes(); starts = [0.0]
    for s in S: starts.append(starts[-1] + s.dur)
    total = starts[-1]
    if cmd == "info":
        print(f"{len(S)} scenes, total {total:.1f} s")
        for i, s in enumerate(S): print(i, type(s).__name__, s.section, s.dur)
    elif cmd == "still":
        i, t = int(sys.argv[2]), float(sys.argv[3]); img = S[i].frame(t).convert("RGB"); chrome(img, S[i].section, starts[i], t, total)
        img.save(f"still_s{i}_{t}.jpg", quality=88)
    elif cmd == "render":
        for i in map(int, sys.argv[2:]):
            render_scene(S[i], starts[i], total, os.path.join(PARTS, f"scene_{i:02d}.mp4")); print("done", i, flush=True)
    elif cmd == "concat":
        lst = os.path.join(PARTS, "list.txt")
        open(lst, "w").write("".join(f"file 'scene_{i:02d}.mp4'\n" for i in range(len(S))))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", "-movflags", "+faststart", sys.argv[2]], check=True)
