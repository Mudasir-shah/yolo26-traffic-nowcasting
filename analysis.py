"""Counting, analytics and the forecasting method (Horizon-aware Conformal Flow Nowcasting).

Idea
----
1. Inverse-depth time-to-line (TTL) forecast. On a flat road seen by a pinhole camera, the image row y of a
   vehicle's ground-contact point satisfies (y - y_h) ∝ 1/D, where y_h is the horizon row and D the distance.
   At constant ground speed, u = 1/(y - y_h) is therefore *linear in time*. We estimate y_h from the tracks
   themselves (no camera calibration) and extrapolate u linearly to the counting line.
2. Online conformal intervals. Normalised residuals of past forecasts (only vehicles that have already crossed,
   so there is no look-ahead) give an 80% interval for every new forecast.
3. Probabilistic flow nowcast. Each approaching vehicle's conformal predictive distribution gives P(cross within
   H s). Summing them (Poisson-binomial), plus a rate term for vehicles not yet in view, gives the expected
   number of line crossings in the next H seconds with an 80% interval.
Baselines: linear extrapolation of image y (TTL) and persistence (count in the previous H s) for the nowcast.
"""
import json, collections, math
import numpy as np

VEHICLE = {"car", "truck", "bus", "motorcycle", "bicycle"}


class Analysis:
    def __init__(self, tracks_path, line, H_now=3.0, window_s=0.6, alpha=0.2, static_px=None, split=0.4, gamma=0.01):
        d = json.load(open(tracks_path))
        self.fps = d["fps"]; self.frames = d["frames"]; self.N = len(self.frames)
        names = {int(k): v for k, v in d["names"].items()}
        self.LY, self.LX0, self.LX1 = line
        self.H_now, self.alpha, self.gamma = H_now, alpha, gamma
        self.split_frame = int(split * len(self.frames))
        self.win = max(4, int(round(window_s * self.fps)))
        self.hist = collections.defaultdict(dict); votes = collections.defaultdict(collections.Counter)
        for fi, dets in enumerate(self.frames):
            for tid, c, s, x1, y1, x2, y2 in dets:
                self.hist[tid][fi] = ((x1 + x2) / 2, y2, y2 - y1); votes[tid][names[c]] += s
        self.cls = {t: v.most_common(1)[0][0] for t, v in votes.items()}
        self.first = {t: min(h) for t, h in self.hist.items()}
        hh = np.median([h[f][2] for t, h in self.hist.items() for f in h])
        self.static_px = static_px or max(3.0, 0.08 * hh)
        self._events(); self._horizon(); self._forecasts(); self._nowcast()

    # ---------------- counting ----------------
    def _events(self):
        self.events = {}
        for tid, h in self.hist.items():
            fs = sorted(h)
            for a, b in zip(fs, fs[1:]):
                (xa, ya, _), (xb, yb, _) = h[a], h[b]
                if self.LX0 <= xb <= self.LX1 and (ya < self.LY <= yb or yb < self.LY <= ya) and b - a <= 5:
                    self.events[tid] = (b, "in" if yb > ya else "out"); break
        self.ev_frames = sorted((f, t) for t, (f, _) in self.events.items())

    def moving(self, tid, fi):
        h = self.hist[tid]; past = h.get(fi - int(self.fps)) or h[min(h)]
        return abs(h[fi][0] - past[0]) + abs(h[fi][1] - past[1]) >= self.static_px

    # ---------------- horizon estimate ----------------
    def _segments(self, n=None):
        segs = []
        for tid, (cf, _) in self.events.items():
            if cf >= self.split_frame: continue
            h = self.hist[tid]
            fs = [f for f in range(cf - int(2 * self.fps), cf + 1) if f in h]
            if len(fs) >= self.win:
                segs.append((np.array(fs) / self.fps, np.array([h[f][1] for f in fs])))
        return segs

    def _horizon(self):
        segs = self._segments()
        ymin = min(y.min() for _, y in segs) if segs else self.LY
        best = (1e9, ymin - 50)
        for yh in np.linspace(ymin - 3 * (self.LY + 1), ymin - 2, 300):
            err = 0.0
            for t, y in segs:
                u = 1.0 / (y - yh); A = np.vstack([t, np.ones_like(t)]).T
                coef, *_ = np.linalg.lstsq(A, u, rcond=None)
                r = u - A @ coef; err += float(np.sum(r ** 2) / (np.var(u) * len(u) + 1e-12))
            if err < best[0]: best = (err, yh)
        self.y_h = float(best[1])

    # ---------------- per-vehicle TTL forecasts ----------------
    def _ttl(self, ts, ys, model):
        t0 = ts[-1]
        if model == "quadratic":
            a2, a1, a0 = np.polyfit(ts - t0, ys, 2)
            roots = np.roots([a2, a1, a0 - self.LY]) if abs(a2) > 1e-9 else ([(self.LY - a0) / a1] if abs(a1) > 1e-9 else [])
            pos = [r.real for r in np.atleast_1d(roots) if abs(np.imag(r)) < 1e-9 and r.real > 0]
            ttl = min(pos) if pos else None
            return ttl if ttl is not None and ttl <= 6.0 else None
        if model == "linear":
            v, b = np.polyfit(ts, ys, 1)
            if abs(v) < 1e-6: return None
            ttl = (self.LY - (v * t0 + b)) / v
        else:
            if np.any(ys - self.y_h <= 1): return None
            u = 1.0 / (ys - self.y_h); k, b = np.polyfit(ts, u, 1)
            uL = 1.0 / (self.LY - self.y_h)
            if abs(k) < 1e-9: return None
            ttl = (uL - (k * t0 + b)) / k
        return ttl if 0.0 < ttl <= 6.0 else None

    def _forecasts(self):
        """For every frame and every moving, not-yet-crossed vehicle heading to the line, forecast TTL."""
        self.fc = collections.defaultdict(dict)   # fi -> tid -> {'lin','inv'}
        self.records = []                          # (fi, tid, lead_true, lin, inv)
        for fi in range(self.N):
            for tid, c, s, *_ in self.frames[fi]:
                if self.cls.get(tid) not in VEHICLE: continue
                h = self.hist[tid]
                if tid in self.events and self.events[tid][0] <= fi: continue
                fs = [f for f in range(fi - self.win + 1, fi + 1) if f in h]
                if len(fs) < self.win - 1 or not self.moving(tid, fi): continue
                x = h[fi][0]
                if not (self.LX0 - 40 <= x <= self.LX1 + 40): continue
                ts = np.array(fs) / self.fps; ys = np.array([h[f][1] for f in fs])
                lin, quad, inv = self._ttl(ts, ys, "linear"), self._ttl(ts, ys, "quadratic"), self._ttl(ts, ys, "inverse")
                if inv is None: continue
                # where will it cross? extrapolate x linearly; skip vehicles that leave the line's extent first
                xs = np.array([h[f][0] for f in fs]); vx, bx = np.polyfit(ts, xs, 1)
                x_cross = vx * (ts[-1] + inv) + bx
                if not (self.LX0 <= x_cross <= self.LX1): continue
                self.fc[fi][tid] = {"lin": lin, "quad": quad, "inv": inv}
                if tid in self.events:
                    self.records.append((fi, tid, (self.events[tid][0] - fi) / self.fps, lin, inv, quad))
        # online adaptive conformal calibration (ACI) on normalised residuals; truths used only once known
        self.cal_ready_frame = None
        recs = sorted(self.records, key=lambda r: self.events[r[1]][0])
        cal, ptr, a_t = [], 0, self.alpha
        self.cov, self.cal_scores_at, self.alpha_trace = [], {}, []
        for fi in range(self.N):
            while ptr < len(recs) and self.events[recs[ptr][1]][0] <= fi:
                f0, tid, true, _, inv, _ = recs[ptr]; p = self.fc[f0][tid]; ptr += 1
                if p.get("calibrated"):
                    miss = not (p["lo"] <= true <= p["hi"])
                    a_t = float(np.clip(a_t + self.gamma * (self.alpha - miss), 0.01, 0.6))
                    if f0 >= self.split_frame: self.cov.append((true, not miss, p["hi"] - p["lo"]))
                cal.append((true - inv) / (inv + 0.25))
            arr = np.array(cal[-400:]) if len(cal) >= 30 else None
            if arr is not None and self.cal_ready_frame is None: self.cal_ready_frame = fi
            self.cal_scores_at[fi] = arr; self.alpha_trace.append(a_t)
            for tid, p in self.fc[fi].items():
                inv = p["inv"]
                if arr is None: p["lo"], p["hi"] = inv * 0.6, inv * 1.4
                else:
                    qlo, qhi = np.quantile(arr, [a_t / 2, 1 - a_t / 2])
                    p["lo"], p["hi"], p["calibrated"] = max(0.0, inv + qlo * (inv + 0.25)), inv + qhi * (inv + 0.25), True

    # ---------------- probabilistic flow nowcast ----------------
    def _nowcast(self):
        H = self.H_now; nH = int(round(H * self.fps))
        self.now = []   # fi, expected, lo80, hi80, persistence, actual
        cross_f = sorted(f for f, _ in self.ev_frames)
        for fi in range(self.N):
            arr = self.cal_scores_at[fi]
            ps = []
            for tid, p in self.fc[fi].items():
                inv = p["inv"]
                if arr is None: ps.append(1.0 if inv <= H else 0.0)
                else: ps.append(float(np.mean(inv + arr * (inv + 0.25) <= H)))
            # rate of crossings by vehicles that were not yet in view H seconds before crossing
            past = [(f, t) for f, t in self.ev_frames if f <= fi]
            # a vehicle first seen L frames before crossing is invisible for (nH - L) of the nH forecast origins
            unseen = sum((nH - (f - self.first[t])) / nH for f, t in past if f - self.first[t] < nH)
            lam = unseen / max(fi / self.fps, 10.0)
            # distribution of total count: Poisson-binomial * Poisson(lam*H)
            dist = np.array([1.0])
            for p in ps: dist = np.convolve(dist, [1 - p, p])
            mu = lam * H; K = 30
            pois = np.array([math.exp(-mu) * mu ** k / math.factorial(k) for k in range(K)])
            dist = np.convolve(dist, pois)[:60]; cdf = np.cumsum(dist)
            exp_ = float(sum(ps) + mu)
            lo = int(np.searchsorted(cdf, 0.1)); hi = int(np.searchsorted(cdf, 0.9))
            persist = sum(1 for f in cross_f if fi - nH < f <= fi)
            actual = sum(1 for f in cross_f if fi < f <= fi + nH)
            self.now.append((fi, exp_, lo, hi, persist, actual))

    # ---------------- summary metrics ----------------
    def metrics(self):
        out = {"y_horizon": round(self.y_h, 1), "crossings": len(self.events),
               "by_class": dict(collections.Counter(self.cls[t] for t in self.events)),
               "by_direction": dict(collections.Counter(d for _, d in self.events.values()))}
        ev = [r for r in self.records if r[0] >= self.split_frame and r[3] is not None and r[5] is not None]
        mae = lambda sel, k: round(float(np.mean([abs(r[k] - r[2]) for r in sel])), 3)
        rows = []
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 6)]:
            sel = [r for r in ev if a < r[2] <= b]
            if sel: rows.append({"lead_s": f"{a}-{b}", "n": len(sel), "linear": mae(sel, 3), "const_accel": mae(sel, 5), "ours_inverse_depth": mae(sel, 4)})
        out["ttl_mae_s"] = rows
        if ev: out["ttl_mae_all_s"] = {"n": len(ev), "linear": mae(ev, 3), "const_accel": mae(ev, 5), "ours_inverse_depth": mae(ev, 4)}
        out["eval_from_s"] = round(self.split_frame / self.fps, 1)
        if self.cov:
            out["conformal"] = {"target": 1 - self.alpha, "coverage": round(float(np.mean([c for _, c, _ in self.cov])), 3),
                                "mean_width_s": round(float(np.mean([w for *_, w in self.cov])), 3), "n": len(self.cov)}
        start = max(self.cal_ready_frame or 0, self.split_frame)
        sel = [r for r in self.now if r[0] >= start and r[0] + self.H_now * self.fps < self.N]
        if sel:
            out["nowcast"] = {"horizon_s": self.H_now,
                              "mae_ours": round(float(np.mean([abs(r[1] - r[5]) for r in sel])), 3),
                              "mae_persistence": round(float(np.mean([abs(r[4] - r[5]) for r in sel])), 3),
                              "interval_coverage": round(float(np.mean([r[2] <= r[5] <= r[3] for r in sel])), 3),
                              "n_frames": len(sel)}
        return out
