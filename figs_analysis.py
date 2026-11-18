"""Run the forecasting analysis on all videos, save metrics and make the result figures."""
import json, os, pickle, collections
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analysis import Analysis
from configs import VIDEOS, FREEWAY_LINE_SWEEP

os.makedirs("figs", exist_ok=True); os.makedirs("results", exist_ok=True)
C_LIN, C_QUAD, C_OURS, C_PERS = "#64748b", "#fbbf24", "#a78bfa", "#64748b"
LIME, CYAN, TXT, GRID = "#a3e635", "#22d3ee", "#e2e8f0", "#23324f"
plt.rcParams.update({"figure.facecolor": "#0d142a", "axes.facecolor": "#0d142a", "axes.edgecolor": "#33476b",
                     "axes.labelcolor": TXT, "xtick.color": TXT, "ytick.color": TXT, "text.color": TXT,
                     "font.size": 15, "axes.grid": True, "grid.color": GRID, "axes.titleweight": "bold",
                     "axes.titlesize": 18, "legend.frameon": False, "font.family": "DejaVu Sans"})

def load(key, line=None):
    v = VIDEOS[key]; line = line or v["line"]
    pk = f"results/cache_{key}_{line[0]}.pkl"
    if os.path.exists(pk): return pickle.load(open(pk, "rb"))
    a = Analysis(v["tracks"], line); pickle.dump(a, open(pk, "wb")); return a

A = {k: load(k) for k in VIDEOS}
M = {k: a.metrics() for k, a in A.items()}
json.dump(M, open("results/forecast_metrics.json", "w"), indent=2)

# 1) TTL error by lead time (pooled, weighted by n)
bins = ["0-1", "1-2", "2-3", "3-6"]; pooled = {b: {"n": 0, "linear": 0, "const_accel": 0, "ours_inverse_depth": 0} for b in bins}
for m in M.values():
    for r in m.get("ttl_mae_s", []):
        p = pooled[r["lead_s"]]; p["n"] += r["n"]
        for k in ("linear", "const_accel", "ours_inverse_depth"): p[k] += r[k] * r["n"]
bins = [b for b in bins if pooled[b]["n"] > 0]
vals = {k: [pooled[b][k] / pooled[b]["n"] for b in bins] for k in ("linear", "const_accel", "ours_inverse_depth")}
fig, ax = plt.subplots(figsize=(11, 6.2)); x = np.arange(len(bins)); w = .26
for i, (k, lab, col) in enumerate([("linear", "Linear extrapolation", C_LIN), ("const_accel", "Constant acceleration", C_QUAD), ("ours_inverse_depth", "Ours: inverse-depth", C_OURS)]):
    bars = ax.bar(x + (i - 1) * w, vals[k], w, label=lab, color=col)
    for b_, v in zip(bars, vals[k]): ax.text(b_.get_x() + b_.get_width() / 2, v + .03, f"{v:.2f}", ha="center", fontsize=12)
ax.set_xticks(x, [f"{b} s\n(n={pooled[b]['n']})" for b in bins]); ax.set_xlabel("Forecast lead time (true seconds until crossing)")
ax.set_ylabel("Mean absolute error (s)"); ax.set_title("Time-to-line forecast error · 3 videos, held-out part"); ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig("figs/fig_ttl_mae.png", dpi=110); plt.close(fig)

# 2) per-video overall TTL MAE
keys = list(VIDEOS); fig, ax = plt.subplots(figsize=(11, 6.2)); x = np.arange(len(keys))
for i, (k, lab, col) in enumerate([("linear", "Linear", C_LIN), ("const_accel", "Const. accel.", C_QUAD), ("ours_inverse_depth", "Ours", C_OURS)]):
    v = [M[kk]["ttl_mae_all_s"][k] for kk in keys]; bars = ax.bar(x + (i - 1) * w, v, w, label=lab, color=col)
    for b_, vv in zip(bars, v): ax.text(b_.get_x() + b_.get_width() / 2, vv + .02, f"{vv:.2f}", ha="center", fontsize=12)
ax.set_xticks(x, [f"{VIDEOS[k]['title']}\n(n={M[k]['ttl_mae_all_s']['n']})" for k in keys], fontsize=13)
ax.set_ylabel("Mean absolute error (s)"); ax.set_title("Time-to-line error per video"); ax.legend()
fig.tight_layout(); fig.savefig("figs/fig_ttl_per_video.png", dpi=110); plt.close(fig)

# 3) conformal coverage + nowcast MAE
fig, axs = plt.subplots(1, 2, figsize=(14, 6.2))
cov = [M[k]["conformal"]["coverage"] for k in keys]; wid = [M[k]["conformal"]["mean_width_s"] for k in keys]
bars = axs[0].bar(x, cov, .55, color=CYAN); axs[0].axhline(.8, color=LIME, ls="--", lw=2); axs[0].text(len(keys) - .55, .815, "target 80%", color=LIME, fontsize=13)
for b_, c, wd in zip(bars, cov, wid): axs[0].text(b_.get_x() + b_.get_width() / 2, c + .015, f"{c:.0%}\nwidth {wd:.2f}s", ha="center", fontsize=12)
axs[0].set_ylim(0, 1.05); axs[0].set_xticks(x, [VIDEOS[k]["title"].replace(" ", "\n", 1) for k in keys], fontsize=12)
axs[0].set_title("Adaptive conformal intervals: coverage")
o = [M[k]["nowcast"]["mae_ours"] for k in keys]; p = [M[k]["nowcast"]["mae_persistence"] for k in keys]
b1 = axs[1].bar(x - .18, p, .36, color=C_PERS, label="Persistence (last 3 s)"); b2 = axs[1].bar(x + .18, o, .36, color=C_OURS, label="Ours: probabilistic nowcast")
for b_, v in list(zip(b1, p)) + list(zip(b2, o)): axs[1].text(b_.get_x() + b_.get_width() / 2, v + .03, f"{v:.2f}", ha="center", fontsize=12)
axs[1].set_xticks(x, [VIDEOS[k]["title"].replace(" ", "\n", 1) for k in keys], fontsize=12); axs[1].set_ylabel("MAE (vehicles per 3 s)")
axs[1].set_title("Next-3 s count forecast error"); axs[1].legend(loc="upper left", fontsize=12)
fig.tight_layout(); fig.savefig("figs/fig_conformal_nowcast.png", dpi=110); plt.close(fig)

# 4) urban nowcast timeline
a = A["urban"]; st = max(a.cal_ready_frame or 0, a.split_frame); en = a.N - int(a.H_now * a.fps)
fr = np.arange(st, en); t = fr / a.fps
e = np.array([a.now[f][1] for f in fr]); lo = np.array([a.now[f][2] for f in fr]); hi = np.array([a.now[f][3] for f in fr])
act = np.array([a.now[f][5] for f in fr]); per = np.array([a.now[f][4] for f in fr])
fig, ax = plt.subplots(figsize=(14, 6))
ax.fill_between(t, lo, hi, color=C_OURS, alpha=.25, step="mid", label="80% interval")
ax.plot(t, e, color=C_OURS, lw=2.5, label="Forecast (expected)"); ax.plot(t, per, color=C_PERS, lw=1.6, ls="--", label="Persistence")
ax.plot(t, act, color=LIME, lw=2.2, label="Realised count")
ax.set_xlabel("Video time (s), held-out part"); ax.set_ylabel("Vehicles crossing in next 3 s"); ax.set_title("Urban road: next-3 s flow forecast vs what happened")
ax.legend(ncol=4, loc="upper left", fontsize=12); ax.set_ylim(0, max(hi.max(), act.max()) + 2)
fig.tight_layout(); fig.savefig("figs/fig_nowcast_timeline.png", dpi=110); plt.close(fig)

# 5) freeway line-position sweep
rows = []
for ly in FREEWAY_LINE_SWEEP:
    aa = load("freeway", (ly, 0, 1280)); mm = aa.metrics()
    lags = [(f - aa.first[tid]) / aa.fps for f, tid in aa.ev_frames]
    rows.append({"line_y": ly, "lead_median_s": round(float(np.median(lags)), 2), **mm["nowcast"]})
json.dump(rows, open("results/freeway_line_sweep.json", "w"), indent=2)
fig, ax = plt.subplots(figsize=(11, 6.2)); x2 = np.arange(len(rows))
b1 = ax.bar(x2 - .18, [r["mae_persistence"] for r in rows], .36, color=C_PERS, label="Persistence")
b2 = ax.bar(x2 + .18, [r["mae_ours"] for r in rows], .36, color=C_OURS, label="Ours")
for b_, r in zip(b2, rows): ax.text(b_.get_x() + b_.get_width() / 2, r["mae_ours"] + .05, f"{r['mae_ours']:.2f}", ha="center", fontsize=12)
for b_, r in zip(b1, rows): ax.text(b_.get_x() + b_.get_width() / 2, r["mae_persistence"] + .05, f"{r['mae_persistence']:.2f}", ha="center", fontsize=12)
ax.set_xticks(x2, [f"line y={r['line_y']}\nvisible {r['lead_median_s']:.1f} s" for r in rows], fontsize=13)
ax.set_ylabel("MAE (vehicles per 3 s)"); ax.set_title("Freeway: forecast skill vs how long cars are visible before the line"); ax.legend()
fig.tight_layout(); fig.savefig("figs/fig_line_sweep.png", dpi=110); plt.close(fig)

# 6) urban trajectory heatmap over a background frame
cap = cv2.VideoCapture(VIDEOS["urban"]["path"]); cap.set(cv2.CAP_PROP_POS_FRAMES, 40); ok, bgf = cap.read()
hh, ww = bgf.shape[:2]; heat = np.zeros((hh, ww), np.float32)
for tid, h in a.hist.items():
    pts = [(int(h[f][0]), int(h[f][1])) for f in sorted(h)]
    if len(pts) > 5 and abs(pts[-1][1] - pts[0][1]) > 40:
        for p0, p1 in zip(pts, pts[1:]): cv2.line(heat, p0, p1, 1.0, 6)
heat = cv2.GaussianBlur(heat, (0, 0), 9); heat /= heat.max() + 1e-6
cm = (plt.get_cmap("magma")(heat)[..., :3] * 255).astype(np.uint8)[..., ::-1]
dim = (bgf * .45).astype(np.uint8); m3 = (np.clip(heat * 1.6, 0, 1))[..., None]
out = (dim * (1 - m3) + cm * m3).astype(np.uint8)
ly = VIDEOS["urban"]["line"][0]; cv2.line(out, (0, ly), (ww, ly), (36, 191, 251), 3)
cv2.imwrite("figs/fig_heatmap.jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(json.dumps(M, indent=1)); print(json.dumps(rows, indent=1))
