"""Fine-tune Ultralytics YOLO26n (COCO-pretrained) on a KITTI subset and validate it.

KITTI subset: 800 train / 200 val images sampled with seed 0 from the Ultralytics KITTI release
(kitti.zip, 5,985 train / 1,496 val, 8 classes). CPU-only training (2 cores), so the subset, image size and epoch count are small.
"""
import json, sys, time
from ultralytics import YOLO

DATA = sys.argv[1] if len(sys.argv) > 1 else "kitti_sub.yaml"
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 20

IMGSZ = int(sys.argv[3]) if len(sys.argv) > 3 else 416
MODEL = sys.argv[4] if len(sys.argv) > 4 else "yolo26n.pt"          # e.g. yolo26s.pt / yolo26m.pt on a GPU
DEVICE = sys.argv[5] if len(sys.argv) > 5 else "cpu"                # "0" for the first CUDA GPU
BATCH = int(sys.argv[6]) if len(sys.argv) > 6 else 16
cfg = dict(data=DATA, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, workers=2 if DEVICE == "cpu" else 8, device=DEVICE, seed=0,
           optimizer="auto", patience=0, project="runs", name="yolo26n_kitti", exist_ok=True, plots=True)
t0 = time.time()
model = YOLO(MODEL)
model.train(**cfg)
train_min = (time.time() - t0) / 60

save_dir = model.trainer.save_dir
best = YOLO(f"{save_dir}/weights/best.pt")
m = best.val(data=DATA, imgsz=IMGSZ, batch=BATCH, device=DEVICE, plots=True, project="runs", name="val_best", exist_ok=True)
per_class = {best.names[c]: {"AP50": round(float(m.box.ap50[i]), 3), "AP50-95": round(float(m.box.ap[i]), 3)}
             for i, c in enumerate(m.box.ap_class_index)}
json.dump({"config": {k: v for k, v in cfg.items() if k not in ("project", "name", "exist_ok", "plots")},
           "train_minutes": round(train_min, 1), "save_dir": str(save_dir), "model": MODEL.replace(".pt", ""),
           "params_m": round(sum(p.numel() for p in best.model.parameters()) / 1e6, 1),
           "hardware": "CPU (2 cores)" if DEVICE == "cpu" else __import__("torch").cuda.get_device_name(int(DEVICE.split(",")[0])),
           "train_images": len(best.trainer.train_loader.dataset) if getattr(best, "trainer", None) else None,
           "metrics": {"precision": round(float(m.box.mp), 3), "recall": round(float(m.box.mr), 3),
                       "mAP50": round(float(m.box.map50), 3), "mAP50-95": round(float(m.box.map), 3)},
           "per_class": per_class, "speed_ms": m.speed}, open("train_summary.json", "w"), indent=2)
print(open("train_summary.json").read())
