import sys, os, time, yaml, logging
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.logging.logger_setup import setup_logger
from src.capture.video_capture import VideoSource
from ultralytics import YOLO

def benchmark(model, video_path, num_frames=100):
    video = VideoSource(source=video_path, reconnect_sec=0.5)
    video.open()
    latencies = []
    for i in range(num_frames):
        frame = video.read()
        if frame is None:
            break
        t0 = time.perf_counter()
        model.predict(frame, verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000)  # ms
    video.close()
    lat = np.array(latencies)
    return {
        "frames": len(lat),
        "avg_ms": float(lat.mean()),
        "p95_ms": float(np.percentile(lat, 95)),
        "max_ms": float(lat.max()),
        "fps": 1000.0 / float(lat.mean())
    }

def main():
    setup_logger()
    logger = logging.getLogger("benchmark")
    video_path = "data/test_video_1.mp4"
    NUM_FRAMES = 100

    for name, weights in [
        ("YOLO26n (PPE)", "models/ppe_yolo26n_best.pt"),
        ("YOLOv8n (COCO)", "models/yolov8n.pt"),
    ]:
        logger.info(f"=== Бенчмарк: {name} ===")
        model = YOLO(weights)
        result = benchmark(model, video_path, NUM_FRAMES)
        logger.info(f"{name}: {result}")

if __name__ == "__main__":
    main()
