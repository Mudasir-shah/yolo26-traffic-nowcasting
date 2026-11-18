"""Build the research demo video (1920x1080, 30 fps) from analysis results, training outputs and figures."""
import json, math, os, subprocess, sys, collections
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from analysis import Analysis

W, H, FPS = 1920, 1080, 30
ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "figs"); PARTS = os.path.join(ROOT, "parts")
GF = "/usr/share/fonts/truetype/google-fonts/"; IF = "/usr/share/fonts/opentype/inter/"
_fc = {}
def F(kind, size):
    path = {"bold": GF + "Poppins-Bold.ttf", "semi": IF + "Inter-SemiBold.otf", "reg": IF + "Inter-Regular.otf",
            "med": IF + "Inter-Medium.otf", "ibold": IF + "Inter-Bold.otf"}[kind]
    if (path, size) not in _fc: _fc[(path, size)] = ImageFont.truetype(path, size)
    return _fc[(path, size)]

BG0 = np.array([6, 10, 24]); BG1 = np.array([15, 24, 52])
CYAN = (34, 211, 238); VIOLET = (167, 139, 250); LIME = (163, 230, 53); AMBER = (251, 191, 36)
PINK = (244, 114, 182); WHITE = (241, 245, 249); MUTED = (148, 163, 184); GREY = (100, 116, 139)
PANEL = (13, 20, 42); EDGE = (36, 58, 92)
CLASS_COL = {"car": CYAN, "truck": VIOLET, "bus": AMBER, "motorcycle": PINK, "bicycle": LIME}
CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle"]

def make_bg():
    y = np.linspace(0, 1, H)[:, None]; x = np.linspace(0, 1, W)[None, :]
    g = 0.55 * x + 0.45 * y
    img = BG0 * (1 - g[..., None]) + BG1 * g[..., None]
    v = np.sqrt((x - .5) ** 2 + (y - .5) ** 2); img *= (1 - .3 * np.clip(v - .3, 0, 1))[..., None]
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)); d = ImageDraw.Draw(im)
    for xx in range(0, W, 64): d.line([(xx, 0), (xx, H)], fill=(14, 21, 42))
    for yy in range(0, H, 64): d.line([(0, yy), (W, yy)], fill=(14, 21, 42))
    return im
BG = make_bg()

def ease(x): x = max(0.0, min(1.0, x)); return 1 - (1 - x) ** 3

def wrap(text, f, maxw):
    out = []
    for para in text.split("\n"):
        cur = ""
        for w_ in para.split():
            t = (cur + " " + w_).strip()
            if f.getlength(t) <= maxw: cur = t
            else: out.append(cur); cur = w_
        out.append(cur)
    return out

def tblock(text, f, fill, maxw=1600, lh=None):
    lines = wrap(text, f, maxw); lh = lh or int(f.size * 1.32)
    w_ = int(max(f.getlength(l) for l in lines)) + 6
    im = Image.new("RGBA", (max(w_, 2), lh * len(lines) + int(f.size * .4)), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    for i, l in enumerate(lines): d.text((0, i * lh), l, font=f, fill=fill)
    return im

def spaced(text, f, fill, sp=4):
    w_ = int(sum(f.getlength(c) + sp for c in text)) + 4
    im = Image.new("RGBA", (w_, int(f.size * 1.5)), (0, 0, 0, 0)); d = ImageDraw.Draw(im); x = 0
    for c in text: d.text((x, 0), c, font=f, fill=fill); x += f.getlength(c) + sp
    return im

def chip(text, col, f=None):
    f = f or F("semi", 26); tw = int(f.getlength(text))
    im = Image.new("RGBA", (tw + 48, 56), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle([1, 1, tw + 46, 54], 27, fill=col + (36,), outline=col + (255,), width=2)
    d.text((24, 12), text, font=f, fill=WHITE); return im

def card(w, h, fill=PANEL, edge=EDGE, r=18):
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); ImageDraw.Draw(im).rounded_rectangle([0, 0, w - 1, h - 1], r, fill=fill + (255,), outline=edge + (255,), width=2)
    return im

def alpha(im, a):
    if a >= .999: return im
    im = im.copy(); im.putalpha(im.getchannel("A").point(lambda v: int(v * a))); return im

def fit_image(path, maxw, maxh, white_card=False, pad=14):
    im = Image.open(path).convert("RGB"); s = min(maxw / im.width, maxh / im.height)
    im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    if not white_card: return im.convert("RGBA")
    c = Image.new("RGBA", (im.width + 2 * pad, im.height + 2 * pad), (0, 0, 0, 0))
    ImageDraw.Draw(c).rounded_rectangle([0, 0, c.width - 1, c.height - 1], 16, fill=(255, 255, 255, 255))
    c.paste(im, (pad, pad)); return c

# ---------------------------------------------------------------- chrome
SECTIONS = []
def chrome(img, section, gstart, t, total):
    d = ImageDraw.Draw(img)
    d.rectangle([64, 52, 88, 56], fill=CYAN)
    d.text((100, 40), "TRAFFIC FLOW NOWCASTING  ·  ULTRALYTICS YOLO26", font=F("semi", 20), fill=MUTED)
    if section:
        tw = F("semi", 20).getlength(section); d.text((W - 64 - tw, 40), section, font=F("semi", 20), fill=VIOLET)
    frac = (gstart + t) / total
    d.rectangle([0, H - 6, W, H], fill=(18, 28, 56)); d.rectangle([0, H - 6, int(W * frac), H], fill=CYAN)

# ---------------------------------------------------------------- slide scene
class Slide:
    def __init__(self, dur, section, build):
        self.dur, self.section, self.build = dur, section, build; self.els = None
    def frame(self, t):
        if self.els is None: self.els = self.build()
        img = BG.copy(); out = 1.0 if t < self.dur - .4 else max(0, (self.dur - t) / .4)
        for el in self.els:
            im, x, y, dl = el[:4]; kind = el[4] if len(el) > 4 else "rise"
            e = ease((t - dl) / .6)
            if e <= 0: continue
            a = e * out
            if kind == "wipe":
                wv = int(im.width * ease((t - dl) / 1.6))
                if wv < 2: continue
                piece = alpha(im.crop((0, 0, wv, im.height)), out); img.paste(piece, (x, y), piece)
            else:
                piece = alpha(im, a); img.paste(piece, (x, int(y + (1 - e) * 36)), piece)
        return img

# ---------------------------------------------------------------- demo scene
class Demo:
    def __init__(self, dur, section, title, source, video, an, start_s=0.0):
        self.dur, self.section, self.title, self.source = dur, section, title, source
        self.video, self.a, self.start_s = video, an, start_s
        cap = cv2.VideoCapture(video); self.sw, self.sh = int(cap.get(3)), int(cap.get(4)); cap.release()
        self.s = min(1280 / self.sw, 720 / self.sh)
        self.vw, self.vh = int(self.sw * self.s), int(self.sh * self.s)
        self.vx = 40 + (1280 - self.vw) // 2; self.vy = 150 + (720 - self.vh) // 2
        self.cap = None; self.cur = -1; self.src = None
        self.static = self._static()
    def _static(self):
        img = BG.copy(); d = ImageDraw.Draw(img)
        d.text((40, 92), self.title, font=F("bold", 34), fill=WHITE)
        d.rounded_rectangle([self.vx - 5, self.vy - 5, self.vx + self.vw + 4, self.vy + self.vh + 4], 12, outline=EDGE, width=2)
        d.rounded_rectangle([1352, 145, 1880, 875], 18, fill=PANEL, outline=EDGE, width=2)
        # legend strip
        x = 40; y = 902
        for c in CLASSES:
            d.rounded_rectangle([x, y + 6, x + 16, y + 22], 4, fill=CLASS_COL[c]); d.text((x + 24, y), c, font=F("med", 20), fill=MUTED)
            x += 34 + F("med", 20).getlength(c) + 18
        for col, txt in ((GREY, "parked (not counted)"), (LIME, "counted")):
            d.rectangle([x, y + 6, x + 16, y + 22], outline=col, width=3); d.text((x + 24, y), txt, font=F("med", 20), fill=MUTED)
            x += 34 + F("med", 20).getlength(txt) + 18
        d.rounded_rectangle([40, 944, 40 + 22, 966], 5, fill=VIOLET)
        d.text((72, 942), "forecast tag: seconds until this vehicle crosses the line (80% conformal interval)",
               font=F("med", 20), fill=MUTED)
        d.text((40, 990), f"Video: {self.source}   ·   Detector: {os.environ.get("TRACK_MODEL", "yolo26s.pt")[:-3].upper().replace("YOLO", "YOLO")} (COCO)   ·   Tracker: ByteTrack   ·   Forecast: inverse-depth time-to-line + adaptive conformal",
               font=F("med", 18), fill=GREY)
        return img
    def _read(self, idx):
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.video); self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx); self.cur = idx - 1
        while self.cur < idx:
            ok, fr = self.cap.read()
            if not ok: break
            self.src = fr; self.cur += 1
        return self.src
    def P(self, x, y): return x * self.s, y * self.s
    def frame(self, t):
        a = self.a; fi = min(a.N - 1, int((self.start_s + t) * a.fps))
        src = self._read(fi)
        vid = Image.fromarray(cv2.cvtColor(cv2.resize(src, (self.vw, self.vh), interpolation=cv2.INTER_AREA if self.s < 1 else cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB))
        vd = ImageDraw.Draw(vid, "RGBA")
        flash = any(0 <= fi - f < 8 for f, _ in a.ev_frames)
        lc = LIME if flash else AMBER
        x0, y0 = self.P(a.LX0, a.LY); x1, _ = self.P(a.LX1, a.LY)
        vd.line([(x0, y0), (x1, y0)], fill=lc + (255,), width=6 if flash else 4)
        fl = F("ibold", 16); lab = "COUNTING LINE"; vd.rectangle([x0 + 8, y0 - 26, x0 + 20 + fl.getlength(lab), y0 - 4], fill=lc)
        vd.text((x0 + 14, y0 - 25), lab, font=fl, fill=(8, 12, 24))
        moving = parked = 0
        for tid, c, s, bx1, by1, bx2, by2 in a.frames[fi]:
            cname = a.cls.get(tid, "car"); X1, Y1 = self.P(bx1, by1); X2, Y2 = self.P(bx2, by2)
            counted = tid in a.events and a.events[tid][0] <= fi
            if not counted and not a.moving(tid, fi):
                parked += 1; vd.rectangle([X1, Y1, X2, Y2], outline=GREY + (190,), width=2); continue
            moving += 1
            col = LIME if counted else CLASS_COL.get(cname, CYAN)
            h = a.hist[tid]
            tr = [self.P(h[k][0], h[k][1]) for k in range(fi - int(a.fps), fi + 1) if k in h]
            if len(tr) > 1: vd.line(tr, fill=col + (160,), width=3)
            vd.rectangle([X1, Y1, X2, Y2], outline=col + (255,), width=3); vd.rectangle([X1, Y1, X2, Y2], fill=col + (24,))
            big = (Y2 - Y1) >= 44
            if big:
                lbl = f"{cname} {s:.2f}"; w_ = fl.getlength(lbl) + 10
                vd.rectangle([X1, max(0, Y1 - 21), X1 + w_, max(0, Y1 - 21) + 20], fill=col); vd.text((X1 + 5, max(0, Y1 - 21) + 1), lbl, font=fl, fill=(8, 12, 24))
            p = a.fc[fi].get(tid)
            if p and p["inv"] <= 4.0 and (Y2 - Y1) >= 30:
                tag = f"{p['inv']:.1f}s ({p['lo']:.1f}–{p['hi']:.1f})"; tw = fl.getlength(tag) + 10
                ty = min(self.vh - 21, Y2 + 2)
                vd.rectangle([X1, ty, X1 + tw, ty + 20], fill=VIOLET + (235,)); vd.text((X1 + 5, ty + 1), tag, font=fl, fill=(8, 12, 24))
            if counted and fi - a.events[tid][0] < int(.7 * a.fps):
                k = (fi - a.events[tid][0]) / (.7 * a.fps)
                vd.text((X2 + 6, Y1 - 26 * k), "+1", font=F("bold", 30), fill=LIME + (int(255 * (1 - k)),))
        img = self.static.copy(); img.paste(vid, (self.vx, self.vy)); d = ImageDraw.Draw(img)
        self._panel(d, fi, moving, parked)
        return img
    def _panel(self, d, fi, moving, parked):
        a = self.a; X0 = 1384; y = 170
        done = [(f, t) for f, t in a.ev_frames if f <= fi]
        d.text((X0, y), "CROSSED THE LINE", font=F("semi", 22), fill=VIOLET); y += 30
        last = max([f for f, _ in done], default=-99); pop = max(0.0, 1 - (fi - last) / 10)
        d.text((X0, y - 8 * pop), str(len(done)), font=F("bold", int(92 + 16 * pop)), fill=LIME if pop > 0 else WHITE)
        dirs = collections.Counter(a.events[t][1] for _, t in done)
        d.text((X0 + 200, y + 22), f"toward camera  {dirs.get('in', 0)}", font=F("semi", 22), fill=WHITE)
        d.text((X0 + 200, y + 56), f"away from camera  {dirs.get('out', 0)}", font=F("semi", 22), fill=WHITE)
        y += 128
        cc = collections.Counter(a.cls[t] for _, t in done)
        for i, c in enumerate(CLASSES):
            cx = X0 + (i % 3) * 160; cy = y + (i // 3) * 34
            d.rounded_rectangle([cx, cy + 6, cx + 14, cy + 20], 3, fill=CLASS_COL[c])
            d.text((cx + 22, cy), f"{c} {cc.get(c, 0)}", font=F("med", 20), fill=MUTED)
        y += 80
        span = min(30.0, fi / a.fps); recent = sum(1 for f, _ in done if fi - f <= span * a.fps)
        rate = recent / span * 60 if span >= 3 else 0
        d.text((X0, y), "FLOW RATE", font=F("semi", 20), fill=VIOLET)
        d.text((X0 + 170, y - 4), f"{rate:.0f} veh/min" if span >= 3 else "warming up", font=F("bold", 26), fill=WHITE)
        d.text((X0, y + 36), f"in view: {moving} moving · {parked} parked", font=F("med", 19), fill=MUTED)
        y += 80; d.line([(X0, y), (1848, y)], fill=EDGE, width=2); y += 18
        n = a.now[fi]
        d.text((X0, y), f"NEXT {a.H_now:.0f} s FORECAST", font=F("semi", 22), fill=VIOLET); y += 34
        d.text((X0, y), f"{n[1]:.1f}", font=F("bold", 72), fill=WHITE)
        d.text((X0 + 170, y + 14), "vehicles expected", font=F("semi", 22), fill=WHITE)
        d.text((X0 + 170, y + 46), f"80% interval: {n[2]}–{n[3]}", font=F("med", 20), fill=MUTED)
        y += 104
        # sparkline: last 15 s of forecast band vs realised counts (realised known up to fi - H)
        gx0, gx1, gy0, gy1 = X0, 1848, y + 8, y + 150
        d.rounded_rectangle([gx0 - 8, gy0 - 8, gx1 + 8, gy1 + 30], 10, outline=EDGE, width=1)
        span_f = int(15 * a.fps); f0 = max(0, fi - span_f)
        rng = range(f0, fi + 1, 2)
        ymax = max(4, max(a.now[k][3] for k in rng) + 1, max(a.now[k][5] for k in rng if k <= fi - a.H_now * a.fps) + 1 if fi - a.H_now * a.fps > f0 else 0)
        X = lambda k: gx0 + (k - (fi - span_f)) / span_f * (gx1 - gx0)
        Y = lambda v: gy1 - v / ymax * (gy1 - gy0)
        band = [(X(k), Y(a.now[k][3])) for k in rng] + [(X(k), Y(a.now[k][2])) for k in reversed(rng)]
        if len(band) > 2: d.polygon(band, fill=(70, 60, 130))
        exp_ = [(X(k), Y(a.now[k][1])) for k in rng]
        if len(exp_) > 1: d.line(exp_, fill=VIOLET, width=3)
        real = [(X(k), Y(a.now[k][5])) for k in rng if k <= fi - a.H_now * a.fps]
        if len(real) > 1: d.line(real, fill=LIME, width=3)
        d.text((gx0, gy1 + 6), "forecast ± 80%", font=F("med", 16), fill=VIOLET)
        d.text((gx0 + 150, gy1 + 6), "realised", font=F("med", 16), fill=LIME)
        d.text((gx1 - 90, gy1 + 6), "last 15 s", font=F("med", 16), fill=GREY)
        d.text((X0, gy1 + 44), f"t = {fi / a.fps:05.1f} s", font=F("med", 19), fill=GREY)

# ---------------------------------------------------------------- render
def render_scene(scene, gstart, total, out):
    tmp = out.replace(".mp4", ".part.mp4")
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                          "-i", "-", "-c:v", "libx264", "-crf", "21", "-preset", "veryfast", "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
    n = int(round(scene.dur * FPS))
    for i in range(n):
        t = i / FPS; img = scene.frame(t)
        if img.mode != "RGB": img = img.convert("RGB")
        chrome(img, scene.section, gstart, t, total)
        p.stdin.write(img.tobytes())
    p.stdin.close()
    if p.wait() == 0: os.replace(tmp, out)
    else: raise SystemExit(f"failed {out}")
