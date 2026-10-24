import sys
import os
import cv2
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.logging.logger_setup import setup_logger
from src.capture.video_capture import VideoSource
from src.capture.fps_meter import FPSMeter
from src.inference.detector import Detector
import logging

# Классы COCO
TARGET_CLASSES = {"person", "car", "truck", "bus", "motorcycle"}

def draw_detections(frame, detections):
    for det in detections:
        if det.class_name not in TARGET_CLASSES:
            continue
        # Рисуем прямоугольник и подпись
        cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(frame, label, (det.x1, det.y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

def main():
    setup_logger()
    logger = logging.getLogger("test_detector")

    #  Читаем конфиг
    with open("configs/configuration.yaml", "r") as f:
        config = yaml.safe_load(f)

    source_path = config["camera"]["source"]
    reconnect_sec = config["camera"]["reconnect_sec"]
    model_path = config["models"]["detector_path"]
    conf_threshold = config["models"]["person_confidence"]

    video = VideoSource(source=source_path, reconnect_sec=reconnect_sec)
    detector = Detector(model_path=model_path, confidence=conf_threshold)
    fps_meter = FPSMeter(window=30)

    try:
        video.open()
    except RuntimeError as e:
        logger.critical(f"Не удалось запустить видео: {e}")
        return

    frame_count = 0
    total_detections = 0

    try:
        while frame_count < 150:
            frame = video.read()
            frame_count += 1

            detections = detector.predict(frame)

            target_dets = [d for d in detections if d.class_name in TARGET_CLASSES]
            total_detections += len(target_dets)

            current_fps = fps_meter.tick()

            # Сохраняем размеченные кадры для визуальной проверки
            if frame_count in (10, 50, 100, 150):
                annotated = draw_detections(frame.copy(), detections)
                out_path = f"data/detected_frame_{frame_count}.jpg"
                cv2.imwrite(out_path, annotated)
                logger.info(f"Сохранён размеченный кадр: {out_path} "
                            f"(объектов: {len(target_dets)}, "
                            f"классы: {[d.class_name for d in target_dets]})")

            if frame_count % 30 == 0:
                logger.info(f"Кадр {frame_count}, FPS конвейера: {current_fps:.2f}, "
                            f"всего целевых детекций: {total_detections}")

    except Exception as e:
        logger.error(f"Ошибка во время обработки: {e}")
    finally:
        video.close()
        logger.info(f"Тест завершён. Обработано кадров: {frame_count}, "
                    f"всего детекций целевых классов: {total_detections}")

if __name__ == "__main__":
    main()
