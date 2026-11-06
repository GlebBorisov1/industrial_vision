import cv2
import yaml
from ultralytics import YOLO

with open("configs/configuration.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

model = YOLO(cfg["models"]["fire_model_path"])
print("=== Классы модели ===")
print(model.names)  # dict: {id: name}
print(f"Ожидаемый порог fire_confidence: {cfg['models'].get('fire_confidence')}")
print(f"Ожидаемый порог smoke_confidence: {cfg['models'].get('smoke_confidence')}")
print()

cap = cv2.VideoCapture(cfg["camera"]["source"])
for i in range(60):
    ok, frame = cap.read()
    if not ok:
        break
    results = model.predict(frame, conf=0.10, verbose=False)[0]
    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            cls = int(box.cls.item())
            conf = float(box.conf.item())
            name = results.names[cls]
            print(f"Frame {i}: class_id={cls} name='{name}' conf={conf:.3f}")
cap.release()
