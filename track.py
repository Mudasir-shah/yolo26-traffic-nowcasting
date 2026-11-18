"""Step 1: run Ultralytics YOLO26 + ByteTrack on the first 30 s of the video and save tracks."""
import json, sys, time
import cv2
from ultralytics import YOLO

SRC = sys.argv[1] if len(sys.argv) > 1 else "data/az_input.mp4"
OUT = sys.argv[2] if len(sys.argv) > 2 else "tracks.json"
SECONDS = float(sys.argv[3]) if len(sys.argv) > 3 else 30
VEHICLES = [1, 2, 3, 5, 7]  # COCO: bicycle, car, motorcycle, bus, truck

import os
model = YOLO(os.environ.get("TRACK_MODEL", "yolo26s.pt"))   # e.g. TRACK_MODEL=yolo26m.pt on a GPU
cap = cv2.VideoCapture(SRC)
fps = cap.get(cv2.CAP_PROP_FPS)
n = int(round(fps * SECONDS))
frames, t0 = [], time.time()
for i in range(n):
    ok, frame = cap.read()
    if not ok:
        break
    r = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=VEHICLES,
                    conf=0.25, imgsz=640, verbose=False)[0]
    dets = []
    if r.boxes.id is not None:
        for (x1, y1, x2, y2), tid, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.id.int().tolist(),
                                                r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
            dets.append([tid, c, round(s, 3), round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)])
    frames.append(dets)
    if i % 120 == 0:
        print(f"frame {i}/{n}  {time.time() - t0:.0f}s", flush=True)
json.dump({"fps": fps, "names": model.names, "frames": frames}, open(OUT, "w"))
print("done", len(frames), "frames in", round(time.time() - t0), "s")
